"""Регрессия по датам, границам окон и оптимизированной SQL-загрузке."""
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
import calculate_batch_features as m


@pytest.fixture
def tables():
    return {
        'customers': pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'signup_date': ['2024-01-01', '2024-01-01', '2025-08-15', '2025-09-15', '2025-10-15'],
        }),
        'sessions': pd.DataFrame({
            'session_id': [10, 11, 12, 13], 'customer_id': [1, 1, 2, 1],
            'start_time': ['2025-07-30 23:00:00', '2025-08-25 00:00:00', '2025-08-20 00:00:00', '2025-09-01 00:00:00'],
        }),
        'events': pd.DataFrame({
            'event_id': [1, 2, 3, 4, 5, 6], 'session_id': [10, 10, 11, 11, 11, 13],
            'timestamp': ['2025-08-02 00:00:00', '2025-08-01 23:59:59', '2025-08-25 00:00:00', '2025-08-25 00:05:00', '2025-09-01 00:00:00', '2025-09-01 00:01:00'],
            'event_type': ['page_view', 'page_view', 'page_view', 'add_to_cart', 'purchase', 'purchase'],
            'product_id': [7, 8, 9, 9, None, None],
        }),
        'orders': pd.DataFrame({
            'order_id': [1, 2, 3, 4], 'customer_id': [1, 1, 1, 4],
            'order_time': ['2025-01-01 00:00:00', '2025-08-02 00:00:00', '2025-09-01 00:00:00', '2025-09-20 00:00:00'],
            'total_usd': [100., 40., 999., 70.],
        }),
    }


def assert_equal(left, right):
    left = left.copy(); right = right.copy()
    for df in (left, right):
        df['run_date'] = df['run_date'].astype('datetime64[ns, UTC]')
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('date,n', [('2025-08-01', 2), ('2025-09-01', 3), ('2025-10-01', 4), ('2025-11-01', 5)])
def test_users_depend_only_on_date(tables, date, n):
    out = m.build_batch_features(tables, date)
    assert len(out) == n
    assert out.customer_id.tolist() == list(range(1, n + 1))


def test_window_boundaries_and_old_session(tables):
    out = m.build_batch_features(tables, '2025-09-01').set_index('customer_id')
    # Событие на нижней границе входит, предыдущее и события на T не входят.
    assert out.loc[1, 'page_view_count_30d'] == 2
    assert out.loc[1, 'page_view_count_7d'] == 1
    assert out.loc[1, 'session_count_30d'] == 1
    assert out.loc[1, 'avg_session_duration_30d'] == 300
    assert out.loc[2, 'avg_session_duration_30d'] == 0
    assert out.loc[1, 'total_spend_30d'] == 40
    assert out.loc[1, 'cart_to_purchase_conversion_7d'] == 0
    assert out.loc[2, 'days_since_last_purchase'] == -1


def test_future_data_does_not_change_result(tables):
    baseline = m.build_batch_features(tables, '2025-09-01')
    for key, column in [('events', 'timestamp'), ('sessions', 'start_time'), ('orders', 'order_time')]:
        tables[key] = tables[key].loc[pd.to_datetime(tables[key][column]) < pd.Timestamp('2025-09-01')].copy()
    assert_equal(baseline, m.build_batch_features(tables, '2025-09-01'))


def test_old_purchase_survives_window(tables):
    out = m.build_batch_features(tables, '2025-11-01').set_index('customer_id')
    assert out.loc[1, 'order_count_30d'] == 0
    assert out.loc[1, 'days_since_last_purchase'] == 61


def test_duplicates_invalid_dates_and_amounts(tables):
    tables['events'] = pd.concat([tables['events'], tables['events'].iloc[[0]]], ignore_index=True)
    tables['orders']['total_usd'] = tables['orders']['total_usd'].astype(object)
    tables['orders'].loc[1, 'total_usd'] = 'invalid'
    bad = tables['events'].iloc[[0]].copy()
    bad['event_id'] = 500; bad['timestamp'] = 'invalid'
    tables['events'] = pd.concat([tables['events'], bad], ignore_index=True)
    out = m.build_batch_features(tables, '2025-09-01').set_index('customer_id')
    assert out.loc[1, 'page_view_count_30d'] == 2
    assert out.loc[1, 'total_spend_30d'] == 0
    assert np.isfinite(out[m.FEATURE_COLUMNS].to_numpy()).all()


@pytest.mark.parametrize('date', ['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01'])
def test_sql_window_matches_full_history(tables, date, tmp_path):
    # Запросы ANSI SQL проверяются на SQLite; PostgreSQL/SSL — в учебном Airflow.
    db = create_engine('sqlite:///' + str(tmp_path / 'source.db'))
    for name, df in tables.items():
        stored = df.copy()
        for column in ['timestamp', 'start_time', 'order_time', 'signup_date']:
            if column in stored:
                stored[column] = pd.to_datetime(stored[column]).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        stored.to_sql(name, db, index=False)
    with patch.object(m, 'create_engine', return_value=db):
        loaded = m.load_source_tables('unused', date, schema='main')
    assert_equal(m.build_batch_features(tables, date), m.build_batch_features(loaded, date))
    assert loaded['events'].shape[0] <= tables['events'].shape[0]
    assert 'last_orders' in loaded


def test_safe_ratio():
    np.testing.assert_array_equal(m.safe_ratio([3, 0, 2], [0, 0, 4]), [0, 0, .5])
