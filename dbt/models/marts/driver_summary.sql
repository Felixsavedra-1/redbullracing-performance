select
    d.driver_name,
    string_agg(distinct con.constructor_name, ', ') as team,
    count(distinct r.season)                         as seasons,
    count(*)                                         as races,
    sum(res.points)                                  as points,
    count(*) filter (where res.position = 1)         as wins,
    count(*) filter (where res.position <= 3)        as podiums,
    round(avg(res.position_order) filter (where not res.is_dnf), 1) as avg_finish,
    count(*) filter (where res.is_dnf)               as dnfs,
    min(r.season)                                    as from_season,
    max(r.season)                                    as to_season
from {{ ref('stg_results') }} res
join {{ ref('stg_races') }}   r   on res.race_id        = r.race_id
join {{ ref('stg_drivers') }} d   on res.driver_id      = d.driver_id
join {{ source('f1_raw', 'constructors') }} con
                                  on res.constructor_id = con.constructor_id
where res.constructor_id = {{ var('constructor_id', 9) }}
group by d.driver_id, d.driver_name
order by points desc
