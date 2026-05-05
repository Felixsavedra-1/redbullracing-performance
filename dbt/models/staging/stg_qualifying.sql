select
    race_id,
    driver_id,
    constructor_id,
    number,
    position,
    q1,
    q2,
    q3
from {{ source('f1_raw', 'qualifying') }}
