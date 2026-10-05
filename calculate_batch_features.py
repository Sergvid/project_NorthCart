"""Подготовка batch-признаков NorthCart относительно run_date."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


FEATURE_COLUMNS = [
    'page_view_count_7d',
    'page_view_count_30d',
    'add_to_cart_count_7d',
    'add_to_cart_count_30d',
    'view_to_cart_conversion_7d',
    'view_to_cart_conversion_30d',
    'cart_to_purchase_conversion_7d',
    'cart_to_purchase_conversion_30d',
    'unique_products_7d',
    'unique_products_30d',
    'avg_session_duration_30d',
    'session_count_7d',
    'session_count_30d',
    'days_since_last_purchase',
    'order_count_30d',
    'total_spend_30d',
    'avg_order_value_30d',
]

OUTPUT_COLUMNS = ['customer_id', 'run_date'] + FEATURE_COLUMNS

# Рабочее допущение: время без часового пояса в источнике означает UTC.
SOURCE_TIMEZONE = 'UTC'


def normalize_run_date(run_date):
    """Возвращает дату среза в UTC."""
    result = pd.Timestamp(run_date)

    if pd.isna(result):
        raise ValueError('run_date не может быть пустой датой')

    if result.tzinfo is None:
        result = result.tz_localize('UTC')
    else:
        result = result.tz_convert('UTC')

    return result


def normalize_datetime(series, source_timezone=SOURCE_TIMEZONE):
    """Приводит время к UTC; невалидные значения заменяет на NaT."""
    parsed = pd.to_datetime(series, errors='coerce')

    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            source_timezone,
            ambiguous='NaT',
            nonexistent='NaT',
        )

    return parsed.dt.tz_convert('UTC')


def safe_ratio(numerator, denominator):
    """При нулевом знаменателе возвращает 0.0."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def load_source_tables(postgres_uri, run_date=None, schema='public', full_history=False):
    """Читает окна 30 дней и последние заказы; full_history нужен только для EDA."""
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', schema):
        raise ValueError('Недопустимое имя схемы PostgreSQL')
    if run_date is None and not full_history:
        raise ValueError('Для оконной загрузки требуется run_date')

    prefix = f'"{schema}".'
    params = {}
    if full_history:
        queries = {
            'customers': f'SELECT customer_id, signup_date FROM {prefix}customers',
            'sessions': f'SELECT session_id, customer_id, start_time FROM {prefix}sessions',
            'events': f'SELECT event_id, session_id, timestamp, event_type, product_id FROM {prefix}events',
            'orders': f'SELECT order_id, customer_id, order_time, total_usd FROM {prefix}orders',
        }
    else:
        cutoff = normalize_run_date(run_date).tz_convert(SOURCE_TIMEZONE).tz_localize(None)
        params = {
            'cutoff': cutoff.to_pydatetime(),
            'window_start': (cutoff - pd.Timedelta(days=30)).to_pydatetime(),
        }
        queries = {
            'customers': f"""
                SELECT customer_id, signup_date FROM {prefix}customers
                WHERE signup_date < :cutoff
            """,
            # Старые сессии нужны только как связь для событий в окне.
            # В счётчики и среднюю длительность они не входят.
            'sessions': f"""
                SELECT s.session_id, s.customer_id, s.start_time
                FROM {prefix}sessions s
                WHERE s.start_time < :cutoff
                  AND (s.start_time >= :window_start OR EXISTS (
                      SELECT 1 FROM {prefix}events e
                      WHERE e.session_id = s.session_id
                        AND e.timestamp >= :window_start
                        AND e.timestamp < :cutoff
                  ))
            """,
            'events': f"""
                SELECT event_id, session_id, timestamp, event_type, product_id
                FROM {prefix}events
                WHERE timestamp >= :window_start AND timestamp < :cutoff
            """,
            'orders': f"""
                SELECT order_id, customer_id, order_time, total_usd
                FROM {prefix}orders
                WHERE order_time >= :window_start AND order_time < :cutoff
            """,
            'last_orders': f"""
                SELECT customer_id, MAX(order_time) AS last_order_time
                FROM {prefix}orders
                WHERE order_time < :cutoff
                GROUP BY customer_id
            """,
        }

    engine = create_engine(
        postgres_uri,
        connect_args={'sslmode': 'require', 'connect_timeout': 15},
        pool_pre_ping=True,
    )
    try:
        tables = {}
        with engine.connect() as connection:
            for name, query in queries.items():
                # Совместимо и с SQLAlchemy 1.4 в Airflow, и с 2.x локально.
                result = connection.execute(text(query), params)
                tables[name] = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
        return tables
    finally:
        engine.dispose()


def preprocess_tables(tables, run_date):
    """Очищает данные и формирует допустимую историю до run_date."""
    run_date = normalize_run_date(run_date)

    specifications = {
        'customers': (
            'signup_date', 'customer_id',
            ['customer_id', 'signup_date'],
        ),
        'sessions': (
            'start_time', 'session_id',
            ['session_id', 'customer_id', 'start_time'],
        ),
        'events': (
            'timestamp', 'event_id',
            ['event_id', 'session_id', 'timestamp'],
        ),
        'orders': (
            'order_time', 'order_id',
            ['order_id', 'customer_id', 'order_time'],
        ),
    }

    cleaned = {}

    for name, (time_column, primary_key, required) in specifications.items():
        df = tables[name].copy()
        df[time_column] = normalize_datetime(df[time_column])
        df = df.dropna(subset=required)
        df = df.loc[df[time_column] < run_date]
        df = df.drop_duplicates(subset=primary_key, keep='first').copy()
        cleaned[name] = df

    customers = cleaned['customers']

    eligible_ids = customers['customer_id']

    sessions = cleaned['sessions'].loc[
        cleaned['sessions']['customer_id'].isin(eligible_ids)
    ].copy()

    orders = cleaned['orders'].loc[
        cleaned['orders']['customer_id'].isin(eligible_ids)
    ].copy()

    orders['total_usd'] = (
        pd.to_numeric(orders['total_usd'], errors='coerce')
        .fillna(0.0)
        .astype(float)
    )

    events = cleaned['events'].merge(
        sessions[['session_id', 'customer_id', 'start_time']],
        on='session_id',
        how='inner',
        validate='many_to_one',
    )

    for df, time_column in [
        (customers, 'signup_date'),
        (sessions, 'start_time'),
        (events, 'timestamp'),
        (orders, 'order_time'),
    ]:
        if not df[time_column].lt(run_date).all():
            raise ValueError(f'Обнаружены будущие данные: {time_column}')

    if 'last_orders' in tables:
        last_orders = tables['last_orders'].copy()
        last_orders['last_order_time'] = normalize_datetime(last_orders['last_order_time'])
        last_orders = last_orders.dropna(subset=['customer_id', 'last_order_time'])
        last_orders = last_orders.loc[
            last_orders['customer_id'].isin(eligible_ids)
            & last_orders['last_order_time'].lt(run_date)
        ]
        last_orders = last_orders.groupby('customer_id', as_index=False)['last_order_time'].max()
    else:
        # Для полного набора из EDA получается тот же результат.
        last_orders = (
            orders.groupby('customer_id')['order_time'].max()
            .rename('last_order_time').reset_index()
        )

    return {
        'customers': customers,
        'sessions': sessions,
        'events': events,
        'orders': orders,
        'last_orders': last_orders,
    }


def add_event_window_features(features, events, run_date):
    """Добавляет счётчики событий и уникальных товаров по двум окнам."""
    features = features.copy()

    for days in [7, 30]:
        start = run_date - pd.Timedelta(days=days)
        window = events.loc[
            events['timestamp'].ge(start)
            & events['timestamp'].lt(run_date)
        ]

        counts = (
            pd.crosstab(window['customer_id'], window['event_type'])
            .reindex(
                columns=['page_view', 'add_to_cart', 'purchase'],
                fill_value=0,
            )
            .reindex(features['customer_id'], fill_value=0)
        )

        for event_type in ['page_view', 'add_to_cart', 'purchase']:
            features[f'{event_type}_count_{days}d'] = (
                counts[event_type].to_numpy(dtype='int64')
            )

        unique_products = (
            window.dropna(subset=['product_id'])
            .groupby('customer_id')['product_id']
            .nunique()
        )

        features[f'unique_products_{days}d'] = (
            features['customer_id']
            .map(unique_products)
            .fillna(0)
            .astype('int64')
        )

    return features


def add_conversion_features(features):
    """Добавляет конверсии по числу событий, без ограничения сверху."""
    features = features.copy()

    for days in [7, 30]:
        features[f'view_to_cart_conversion_{days}d'] = safe_ratio(
            features[f'add_to_cart_count_{days}d'],
            features[f'page_view_count_{days}d'],
        )

        features[f'cart_to_purchase_conversion_{days}d'] = safe_ratio(
            features[f'purchase_count_{days}d'],
            features[f'add_to_cart_count_{days}d'],
        )

    return features


def add_session_features(features, sessions, events, run_date):
    """Добавляет число сессий и среднюю длительность за 30 дней."""
    features = features.copy()

    for days in [7, 30]:
        window = sessions.loc[
            sessions['start_time'].ge(
                run_date - pd.Timedelta(days=days)
            )
            & sessions['start_time'].lt(run_date)
        ]

        counts = window.groupby('customer_id')['session_id'].nunique()

        features[f'session_count_{days}d'] = (
            features['customer_id']
            .map(counts)
            .fillna(0)
            .astype('int64')
        )

    sessions_30d = sessions.loc[
        sessions['start_time'].ge(run_date - pd.Timedelta(days=30))
        & sessions['start_time'].lt(run_date)
    ].copy()

    valid_events = events.loc[
        events['timestamp'].ge(events['start_time'])
        & events['timestamp'].lt(run_date)
    ]

    last_event = (
        valid_events.groupby('session_id')['timestamp']
        .max()
        .rename('last_event_time')
        .reset_index()
    )

    lengths = sessions_30d.merge(
        last_event,
        on='session_id',
        how='left',
        validate='one_to_one',
    )

    lengths['duration_seconds'] = (
        (lengths['last_event_time'] - lengths['start_time'])
        .dt.total_seconds()
        .fillna(0.0)
    )

    mean_duration = (
        lengths.groupby('customer_id')['duration_seconds'].mean()
    )

    features['avg_session_duration_30d'] = (
        features['customer_id']
        .map(mean_duration)
        .fillna(0.0)
        .astype(float)
    )

    return features


def add_order_features(features, orders, run_date, last_orders):
    """Добавляет денежные признаки и давность последнего заказа."""
    features = features.copy()

    window = orders.loc[
        orders['order_time'].ge(run_date - pd.Timedelta(days=30))
        & orders['order_time'].lt(run_date)
    ]

    aggregates = window.groupby('customer_id').agg(
        order_count_30d=('order_id', 'nunique'),
        total_spend_30d=('total_usd', 'sum'),
        avg_order_value_30d=('total_usd', 'mean'),
    )

    features['order_count_30d'] = (
        features['customer_id']
        .map(aggregates['order_count_30d'])
        .fillna(0)
        .astype('int64')
    )

    for column in ['total_spend_30d', 'avg_order_value_30d']:
        features[column] = (
            features['customer_id']
            .map(aggregates[column])
            .fillna(0.0)
            .astype(float)
        )

    last_order = last_orders.set_index('customer_id')['last_order_time']
    last_order_by_customer = features['customer_id'].map(last_order)

    features['days_since_last_purchase'] = (
        (run_date - last_order_by_customer)
        .dt.days
        .fillna(-1)
        .astype('int64')
    )

    return features


def validate_features(features, run_date):
    """Проверяет структуру и основные свойства результата."""
    run_date = normalize_run_date(run_date)

    if features.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError('Неверный состав или порядок колонок')

    if features.empty:
        raise ValueError('Итоговая таблица пуста')

    if features.duplicated(['customer_id', 'run_date']).any():
        raise ValueError('Повторы ключа customer_id, run_date')

    if features.isna().any().any():
        raise ValueError('В итоговой таблице есть пропуски')

    if not features['run_date'].eq(run_date).all():
        raise ValueError('Дата среза не соответствует run_date')

    if not np.isfinite(
        features[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError('В признаках есть бесконечные значения')

    nonnegative = [
        column for column in FEATURE_COLUMNS
        if column != 'days_since_last_purchase'
    ]

    if not features[nonnegative].ge(0).all().all():
        raise ValueError('Обнаружены отрицательные признаки')

    if not features['days_since_last_purchase'].ge(-1).all():
        raise ValueError('Некорректная давность покупки')

    for prefix in [
        'page_view_count',
        'add_to_cart_count',
        'unique_products',
        'session_count',
    ]:
        if not features[f'{prefix}_7d'].le(
            features[f'{prefix}_30d']
        ).all():
            raise ValueError(f'Нарушена вложенность окон: {prefix}')

    for days in [7, 30]:
        for denominator, conversion in [
            ('page_view_count', 'view_to_cart_conversion'),
            ('add_to_cart_count', 'cart_to_purchase_conversion'),
        ]:
            mask = features[f'{denominator}_{days}d'].eq(0)
            if not features.loc[
                mask, f'{conversion}_{days}d'
            ].eq(0.0).all():
                raise ValueError('Неверная обработка деления на ноль')

    no_orders = features['order_count_30d'].eq(0)
    if not features.loc[
        no_orders, ['total_spend_30d', 'avg_order_value_30d']
    ].eq(0.0).all().all():
        raise ValueError('Некорректные суммы при отсутствии заказов')


def build_batch_features(tables, run_date):
    """Собирает 17 признаков, сохраняя одну строку на пользователя."""
    run_date = normalize_run_date(run_date)
    prepared = preprocess_tables(tables, run_date)

    features = (
        prepared['customers'][['customer_id']]
        .sort_values('customer_id')
        .reset_index(drop=True)
    )
    features['run_date'] = run_date

    features = add_event_window_features(
        features, prepared['events'], run_date
    )
    features = add_conversion_features(features)
    features = add_session_features(
        features,
        prepared['sessions'],
        prepared['events'],
        run_date,
    )
    features = add_order_features(
        features, prepared['orders'], run_date, prepared['last_orders']
    )

    # Временные счётчики purchase не входят в итоговые 17 признаков.
    features = features[OUTPUT_COLUMNS].copy()

    validate_features(features, run_date)
    return features


def save_features(features, local_path):
    """Сохраняет CSV локально перед загрузкой в S3."""
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Файл признаков не сохранён или пуст')

    return path
