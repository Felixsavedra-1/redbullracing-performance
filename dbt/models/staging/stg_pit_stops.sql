select
    race_id,
    driver_id,
    stop,
    lap,
    time_of_day,
    duration,
    milliseconds
from {{ source('f1_raw', 'pit_stops') }}
