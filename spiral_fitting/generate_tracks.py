import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def abspath_if_local(value):
    """Lokale Pfade absolut machen, URLs (s3://...) unverändert lassen."""
    return value if '://' in value else str(Path(value).resolve())


def run(cmd, cwd, env):
    print('\n$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main():
    p = argparse.ArgumentParser(
        description='Tracks aus einer Surface Prediction erzeugen '
                    '(extract_surface_tracks, danach build_track_crossings).')
    p.add_argument('--predictions', required=True,
                   help='Thresholded Surface-Prediction (zarr, 3D ZYX)')
    p.add_argument('--out', required=True,
                   help='Ziel-DBM, z. B. /pfad/dataset/tracks/tracks.dbm')
    p.add_argument('--z-min', type=int, required=True)
    p.add_argument('--z-max', type=int, required=True)
    p.add_argument('--backend', default=None)
    p.add_argument('--gpus', default=None)
    p.add_argument('--skip-crossings', action='store_true')
    p.add_argument('--package-dir', default=str(ROOT / 'villa_8aa507c'),
                   help='Ordner des Python-Pakets, in dem extract_surface_tracks.py '
                        'liegt (Standard: ./villa_8aa507c)')
    args = p.parse_args()

    package_dir = Path(args.package_dir).resolve()
    package = package_dir.name
    parent = package_dir.parent

    if not package.isidentifier():
        sys.exit(f"'{package}' ist kein gültiger Paketname (z. B. führender Punkt). "
                 'Benenne den Ordner um oder gib mit --package-dir den richtigen an.')

    scripts = ['extract_surface_tracks'] + ([] if args.skip_crossings else ['build_track_crossings'])
    missing = [str(package_dir / f'{s}.py') for s in scripts
               if not (package_dir / f'{s}.py').is_file()]
    if missing:
        sys.exit('Fehlende Dateien:\n  ' + '\n  '.join(missing))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Wie in deinem Prediction-Skript: Elternordner des Pakets in den PYTHONPATH,
    # damit relative Imports (from .data.utils ...) funktionieren. Der Paketordner
    # selbst kommt dazu, falls die Skripte untereinander flach importieren.
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        filter(None, (str(parent), str(package_dir), env.get('PYTHONPATH'))))

    cmd = [sys.executable, '-m', f'{package}.extract_surface_tracks',
           '--predictions', abspath_if_local(args.predictions),
           '--out', str(out),
           '--z-min', str(args.z_min),
           '--z-max', str(args.z_max)]
    if args.backend:
        cmd += ['--backend', args.backend]
    if args.gpus:
        cmd += ['--gpus', args.gpus]
    run(cmd, parent, env)

    if not args.skip_crossings:
        run([sys.executable, '-m', f'{package}.build_track_crossings', str(out)], parent, env)

    print('\nFertig. Erzeugt:')
    for name in (out.name, out.name + '.vctracks', out.name + '.crossings.npz'):
        path = out.parent / name
        print(f'  {path}  [{"ok" if path.exists() else "nicht gefunden"}]')


if __name__ == '__main__':
    main()