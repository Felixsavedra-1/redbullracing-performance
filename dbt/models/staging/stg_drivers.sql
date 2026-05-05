select
    driver_id,
    driver_ref,
    forename || ' ' || surname as driver_name,
    nationality,
    dob
from {{ source('f1_raw', 'drivers') }}
