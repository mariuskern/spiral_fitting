import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(
        description='Graph Patches (tifxyz) automatisch aus Tracks wachsen lassen '
                    '(grow_track_graph).',
        epilog='Weitere Optionen für grow_track_graph können nach -- angegeben werden.')
    p.add_argument('--tracks', required=True,
                   help='Pfad zur Track-Datenbank, z. B. /pfad/dataset/tracks/tracks.dbm '
                        '(die Sidecars .vctracks und .crossings.npz werden daraus abgeleitet)')
    p.add_argument('--out', required=True,
                   help='Zielordner für die Patches, z. B. /pfad/dataset/graph_patches')
    p.add_argument('--random-count', type=int, default=None,
                   help='Anzahl Patches, die von zufälligen langen Seed-Tracks aus wachsen')
    p.add_argument('--seeds', default=None,
                   help='Alternativ: explizite Track-Zeilen als Seeds '
                        '(Format siehe grow_track_graph --help)')
    p.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1),
                   help='Anzahl paralleler Worker (Standard: min(16, CPU-Kerne))')
    p.add_argument('--package-dir', default=str(ROOT / 'villa_8aa507c'),
                   help='Ordner des Python-Pakets mit grow_track_graph.py '
                        '(Standard: ./villa_8aa507c)')
    p.add_argument('extra', nargs=argparse.REMAINDER,
                   help='Weitere Optionen für grow_track_graph nach --')
    args = p.parse_args()

    if args.random_count is None and args.seeds is None:
        p.error('Entweder --random-count oder --seeds angeben.')
    if args.random_count is not None and args.random_count < 1:
        p.error('--random-count muss mindestens 1 sein.')
    if args.workers < 1:
        p.error('--workers muss mindestens 1 sein.')

    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    if extra and args.extra[:1] != ['--']:
        p.error('Zusätzliche Optionen für grow_track_graph bitte nach -- angeben.')

    package_dir = Path(args.package_dir).resolve()
    package = package_dir.name
    parent = package_dir.parent

    if not package.isidentifier():
        sys.exit(f"'{package}' ist kein gültiger Paketname (z. B. führender Punkt). "
                 'Gib mit --package-dir den Ordner ohne Punkt an.')
    if not (package_dir / 'grow_track_graph.py').is_file():
        sys.exit(f'Fehlende Datei: {package_dir / "grow_track_graph.py"}')

    dbm = Path(args.tracks).resolve()
    vctracks = dbm.parent / (dbm.name + '.vctracks')
    crossings = dbm.parent / (dbm.name + '.crossings.npz')
    missing = [str(f) for f in (vctracks, crossings) if not f.exists()]
    if missing:
        sys.exit('Fehlende Track-Sidecars (zuerst generate_tracks.py ohne '
                 '--skip-crossings ausführen):\n  ' + '\n  '.join(missing))

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        filter(None, (str(parent), str(package_dir), env.get('PYTHONPATH'))))

    cmd = [sys.executable, '-m', f'{package}.grow_track_graph',
           str(vctracks), str(crossings), str(out),
           '--workers', str(args.workers)]
    if args.random_count is not None:
        cmd += ['--random-count', str(args.random_count)]
    if args.seeds is not None:
        cmd += ['--seeds', args.seeds]
    cmd += extra

    print('$ ' + ' '.join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=parent, env=env, check=True)
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)

    count = sum(1 for entry in out.iterdir() if entry.is_dir())
    print(f'\nFertig. {count} Patch-Ordner in {out}')
    print('Im Fit nutzen: in spiral-scroll.json "paths": {"verified_patches": '
          f'"{out.name}"}} setzen (relativ zum Dataset-Ordner) '
          'und input_use_verified_patches aktiv lassen.')


if __name__ == '__main__':
    main()