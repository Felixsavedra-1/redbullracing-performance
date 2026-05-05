select
    r.season,
    d.driver_name,
    count(*)                             as total_stops,
    min(ps.milliseconds)                 as fastest_stop_ms,
    round(avg(ps.milliseconds), 0)       as avg_stop_ms,
    max(ps.milliseconds)                 as slowest_stop_ms,
    round(stddev(ps.milliseconds), 0)    as stddev_stop_ms
from {{ ref('stg_pit_stops') }} ps
join {{ ref('stg_races') }}     r   on ps.race_id  = r.race_id
join {{ ref('stg_drivers') }}   d   on ps.driver_id = d.driver_id
join {{ source('f1_raw', 'results') }} res
    on ps.race_id = res.race_id and ps.driver_id = res.driver_id
where res.constructor_id = {{ var('constructor_id', 9) }}
group by r.season, d.driver_id, d.driver_name
order by r.season desc, avg_stop_ms
