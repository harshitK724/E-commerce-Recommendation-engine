"""PostgreSQL DDL and transform SQL for the interaction pipeline."""

from __future__ import annotations

CREATE_RAW_EVENTS = """
CREATE TABLE IF NOT EXISTS raw_events (
    event_time timestamptz,
    event_type text,
    product_id bigint,
    category_id bigint,
    category_code text,
    brand text,
    price double precision,
    user_id bigint,
    user_session text
);
"""

CREATE_SERVING_TABLES = """
CREATE TABLE IF NOT EXISTS products (
    product_id bigint PRIMARY KEY,
    category_id bigint,
    category_code text,
    brand text,
    price double precision,
    title text
);

CREATE TABLE IF NOT EXISTS user_events (
    user_id bigint NOT NULL,
    product_id bigint NOT NULL,
    event_type text NOT NULL,
    event_time timestamptz,
    weight integer NOT NULL
);

CREATE INDEX IF NOT EXISTS user_events_user_idx ON user_events (user_id);
CREATE INDEX IF NOT EXISTS user_events_product_idx ON user_events (product_id);
"""

TRANSFORM_SQL = """
DROP TABLE IF EXISTS clean_events;
DROP TABLE IF EXISTS user_hour_counts;
DROP TABLE IF EXISTS bot_users;
DROP TABLE IF EXISTS user_counts;
DROP TABLE IF EXISTS item_counts;
DROP TABLE IF EXISTS interactions;

CREATE TABLE clean_events AS
SELECT
    event_time,
    event_type,
    product_id,
    category_id,
    category_code,
    brand,
    price,
    user_id,
    user_session
FROM raw_events
WHERE user_id IS NOT NULL
  AND product_id IS NOT NULL
  AND event_type IN ('view', 'cart', 'purchase');

CREATE TABLE user_hour_counts AS
SELECT
    user_id,
    date_trunc('hour', event_time) AS hour_bucket,
    COUNT(*) AS n_events
FROM clean_events
GROUP BY user_id, date_trunc('hour', event_time);

CREATE TABLE bot_users AS
SELECT DISTINCT user_id
FROM user_hour_counts
WHERE n_events > {bot_events_per_hour};

CREATE TABLE user_counts AS
SELECT user_id, COUNT(*) AS n
FROM clean_events
WHERE user_id NOT IN (SELECT user_id FROM bot_users)
GROUP BY user_id;

CREATE TABLE item_counts AS
SELECT product_id, COUNT(*) AS n
FROM clean_events
WHERE user_id NOT IN (SELECT user_id FROM bot_users)
GROUP BY product_id;

CREATE TABLE interactions AS
SELECT
    c.user_id,
    c.product_id,
    MAX(
        CASE c.event_type
            WHEN 'view' THEN 1
            WHEN 'cart' THEN 3
            WHEN 'purchase' THEN 5
        END
    ) AS weight,
    MAX(c.event_time) AS last_event_time,
    MAX(c.category_id) AS category_id,
    MAX(c.category_code) AS category_code,
    MAX(c.brand) AS brand,
    MAX(c.price) AS price
FROM clean_events c
JOIN user_counts u ON u.user_id = c.user_id AND u.n >= {min_user_events}
JOIN item_counts i ON i.product_id = c.product_id AND i.n >= {min_item_events}
WHERE c.user_id NOT IN (SELECT user_id FROM bot_users)
GROUP BY c.user_id, c.product_id;
"""

PIPELINE_STATS_SQL = """
SELECT
    (SELECT COUNT(*) FROM raw_events) AS raw_rows,
    (SELECT COUNT(*) FROM clean_events) AS clean_rows,
    (SELECT COUNT(*) FROM bot_users) AS bot_users,
    (SELECT COUNT(*) FROM interactions) AS interaction_rows,
    (SELECT COUNT(DISTINCT user_id) FROM interactions) AS n_users,
    (SELECT COUNT(DISTINCT product_id) FROM interactions) AS n_items
"""

COPY_COLUMNS = (
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
)
