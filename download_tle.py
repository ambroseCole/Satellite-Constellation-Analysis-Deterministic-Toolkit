from spacetrack import SpaceTrackClient
import spacetrack.operators as op

st = SpaceTrackClient('YOUR EMAIL', 'YOUR PASSWORD')
debris_tles = st.gp(
    epoch='>now-30',
    # Mean motion range corresponds to 440-660km altitude, adjust for others
    mean_motion=op.inclusive_range(14.5, 16.5),
    orderby='norad_cat_id',
    format='tle'
)

with open('debris_catalog.tle', 'w') as f:
    f.write(debris_tles)

print(f"Saved {len(debris_tles.strip().split(chr(10))) // 2} objects")