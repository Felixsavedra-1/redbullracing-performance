
    
    

select
    driver_name as unique_field,
    count(*) as n_records

from "f1_analytics"."main"."driver_summary"
where driver_name is not null
group by driver_name
having count(*) > 1


