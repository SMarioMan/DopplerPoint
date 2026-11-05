"""Moving source, stationary observer acoustics in Earth-centered coordinates."""
from dataclasses import dataclass
import csv
import numpy as np
from scipy.optimize import least_squares


def ecef(lat, lon, altitude):
    lat, lon = np.radians(lat), np.radians(lon)
    n = 6378137 / np.sqrt(1 - 0.00669437999014 * np.sin(lat)**2)
    return np.array([(n + altitude)*np.cos(lat)*np.cos(lon),
                     (n + altitude)*np.cos(lat)*np.sin(lon),
                     (n*(1-0.00669437999014) + altitude)*np.sin(lat)])


@dataclass
class Track:
    flight_id: str
    time: np.ndarray
    position: np.ndarray
    engine: str = "unknown"

    def __post_init__(self):
        if (len(self.time) < 2 or self.position.shape != (len(self.time), 3)
                or not np.isfinite(self.time).all() or not np.isfinite(self.position).all()
                or np.any(np.diff(self.time) <= 0)):
            raise ValueError("Tracks need at least two finite points with increasing timestamps")
        self.velocity = np.diff(self.position, axis=0) / np.diff(self.time)[:, None]
        if np.any(np.linalg.norm(self.velocity, axis=1) >= 330):
            raise ValueError("Supersonic or discontinuous track; split/filter the input")

    def cpa(self, observer, c=343.):
        delta = self.position[:-1] - observer
        dt = np.clip(-np.sum(delta*self.velocity, axis=1) /
                     np.maximum(np.sum(self.velocity**2, axis=1), 1e-12), 0, np.diff(self.time))
        distance = np.linalg.norm(delta + dt[:, None]*self.velocity, axis=1)
        i = np.argmin(distance)
        t = self.time[i]+dt[i]
        return float(t + distance[i]/c), float(distance[i]), float(np.linalg.norm(self.velocity[i]))

    def curve(self, observer, c=343., step=.5):
        # Sample emission time, then transform every point to its arrival time.
        t = np.arange(self.time[0], self.time[-1], step)
        t = np.unique(np.r_[t, self.time])
        i = np.clip(np.searchsorted(self.time, t, side="right")-1, 0, len(self.time)-2)
        delta = self.position[i] + (t-self.time[i])[:, None]*self.velocity[i] - observer
        distance = np.linalg.norm(delta, axis=1)
        radial = np.sum(delta*self.velocity[i], axis=1)/np.maximum(distance, 1e-9)
        return t+distance/c, c/(c+radial)


def read_tracks(path, max_gap=30.):
    """CSV timestamps are Unix seconds UTC, coordinates degrees, altitude meters."""
    groups = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lat, lon, alt, t = [float(row[k]) for k in ("latitude", "longitude", "altitude_m", "timestamp")]
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Invalid geographic coordinate")
            groups.setdefault(row["flight_id"], []).append((t, ecef(lat, lon, alt), row.get("engine", "unknown")))
    tracks = []
    for name, rows in groups.items():
        rows.sort(key=lambda r: r[0])
        # Duplicate timestamps cannot supply velocity; keep the first sample.
        rows = list({r[0]: r for r in reversed(rows)}.values())
        rows.sort(key=lambda r: r[0])
        chunks = np.split(np.arange(len(rows)), np.where(np.diff([r[0] for r in rows]) > max_gap)[0]+1)
        for j, chunk in enumerate(chunks):
            if len(chunk) < 2:
                continue
            part = [rows[i] for i in chunk]
            tracks.append(Track(f"{name}:{j}", np.array([r[0] for r in part]),
                                np.array([r[1] for r in part]), part[0][2]))
    return tracks


def ideal_curve(arrival_time, f0, speed, distance, arrival_cpa, c=343.):
    # Solve retarded time analytically for a straight, constant-speed source.
    q = np.asarray(arrival_time)-arrival_cpa+distance/c
    a = 1-(speed/c)**2
    emission = (q-np.sqrt((speed*q/c)**2+a*(distance/c)**2))/a
    r = np.sqrt(distance**2+(speed*emission)**2)
    return f0*c/(c+speed**2*emission/r)


def fit_trace(times, frequencies, c=343.):
    times, frequencies = np.asarray(times), np.asarray(frequencies)
    if len(times) < 8 or np.ptp(times) < 5 or not np.isfinite(frequencies).all():
        raise ValueError("Provide at least eight finite trace points spanning five seconds")
    origin = float(np.mean(times))
    result = least_squares(lambda p: ideal_curve(times-origin, *p, c=c)-frequencies,
                           [np.median(frequencies), 80, 1500, 0],
                           bounds=([1, 5, 10, -np.ptp(times)], [20000, min(300, c*.95), 30000, np.ptp(times)]),
                           loss="soft_l1", x_scale="jac")
    f0, speed, distance, t = result.x
    return dict(f0_hz=float(f0), speed_mps=float(speed), slant_m=float(distance),
                audio_cpa_s=float(t+origin), rmse_hz=float(np.sqrt(np.mean(result.fun**2))))
