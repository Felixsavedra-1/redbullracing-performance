with base as (
    select
        r.season,
        r.round,
        r.race_name,
        cs.position  as championship_position,
        cs.points    as points_accumulated,
        cs.wins      as wins_accumulated,
        lag(cs.points) over (partition by r.season order by r.round) as prev_points
    from {{ source('f1_raw', 'constructor_standings') }} cs
    join {{ ref('stg_races') }} r on cs.race_id = r.race_id
    where cs.constructor_id = {{ var('constructor_id', 9) }}
)
select
    *,
    coalesce(points_accumulated - prev_points, points_accumulated) as points_gained
from base
order by season desc, round
