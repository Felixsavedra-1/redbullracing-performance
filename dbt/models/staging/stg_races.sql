select
    race_id,
    year as season,
    round,
    circuit_id,
    race_name,
    race_date,
    race_time
from {{ source('f1_raw', 'races') }}
