select
    driver_id,
    driver_ref,
    forename || ' ' || surname as driver_name,
    nationality,
    dob
from "f1_analytics"."main"."drivers"