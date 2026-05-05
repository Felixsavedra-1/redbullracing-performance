select
    race_id,
    constructor_id,
    points,
    position,
    position_text,
    wins
from {{ source('f1_raw', 'constructor_standings') }}
