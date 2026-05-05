select
    d.driver_name,
    r.season,
    count(*)                                                                          as races_started,
    count(q.position)                                                                 as races_with_qualifying,
    round(avg(q.position), 2)                                                         as avg_qual_position,
    round(avg(res.position_order) filter (where not res.is_dnf), 2)                   as avg_race_position,
    round(avg(cast(res.grid as integer) - res.position_order)
          filter (where not res.is_dnf), 2)                                           as avg_positions_gained,
    count(*) filter (where res.position_order < res.grid and not res.is_dnf)          as races_gained_positions
from "f1_analytics"."main"."stg_results"   res
join "f1_analytics"."main"."stg_races"     r   on res.race_id  = r.race_id
join "f1_analytics"."main"."stg_drivers"   d   on res.driver_id = d.driver_id
left join "f1_analytics"."main"."stg_qualifying" q
    on res.race_id = q.race_id and res.driver_id = q.driver_id
where res.constructor_id = 9
group by d.driver_id, d.driver_name, r.season
order by r.season desc, avg_race_position