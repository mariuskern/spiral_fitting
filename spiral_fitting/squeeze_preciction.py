import argparse
import sys

import numpy as np
import zarr


def open_array(path):
    node = zarr.open(path, mode='r')
    if isinstance(node, zarr.Group):
        keys = list(node.keys())
        print(f'Gruppe mit Einträgen: {keys}')
        if '0' not in keys:
            sys.exit("Gruppe ohne Eintrag '0'; bitte den Array-Pfad direkt angeben.")
        node = node['0']
    return node


def main():
    p = argparse.ArgumentParser(description='Prediction-Zarr prüfen und auf 3D (Z,Y,X) bringen.')
    p.add_argument('prediction', help='Pfad zum Zarr (z. B. .../final_output.zarr)')
    p.add_argument('--out', help='Ziel-Zarr (3D). Ohne Angabe wird nur geprüft.')
    p.add_argument('--channel', type=int, default=0, help='Kanal, der übernommen wird (Standard 0)')
    p.add_argument('--threshold', type=float, default=None,
                   help='Optional: Werte > Schwelle werden 1, sonst 0 (uint8). '
                        'Nur nötig, wenn das Zarr noch nicht binarisiert ist.')
    args = p.parse_args()

    arr = open_array(args.prediction)
    print(f'shape={arr.shape}  dtype={arr.dtype}  chunks={arr.chunks}')

    # Stichprobe: mittlere z-Schicht (letzte drei Achsen sind Z, Y, X)
    lead = (slice(None),) * (arr.ndim - 3)
    mid = arr.shape[-3] // 2
    slab = np.asarray(arr[lead + (mid,)])
    for c in range(slab.shape[0] if arr.ndim == 4 else 1):
        part = slab[c] if arr.ndim == 4 else slab
        values = np.unique(part)
        print(f'Kanal {c}: {len(values)} verschiedene Werte, z. B. {values[:8]}, '
              f'Anteil > 0: {(part > 0).mean():.4f}')

    if not args.out:
        print('\nNur geprüft. Mit --out ein 3D-Zarr schreiben.')
        return
    if arr.ndim == 3:
        sys.exit('Das Array ist bereits 3D; nichts zu tun.')
    if arr.ndim != 4:
        sys.exit(f'{arr.ndim}D wird nicht unterstützt.')
    if not 0 <= args.channel < arr.shape[0]:
        sys.exit(f'--channel muss zwischen 0 und {arr.shape[0] - 1} liegen.')

    shape3 = arr.shape[1:]
    dtype = np.uint8 if args.threshold is not None else arr.dtype
    chunks = tuple(min(c, s) for c, s in zip((64, 256, 256), shape3))
    out = zarr.open(args.out, mode='w', shape=shape3, chunks=chunks, dtype=dtype)

    step = chunks[0] * 4
    for z0 in range(0, shape3[0], step):
        z1 = min(z0 + step, shape3[0])
        block = np.asarray(arr[args.channel, z0:z1])
        if args.threshold is not None:
            block = (block > args.threshold).astype(np.uint8)
        out[z0:z1] = block
        print(f'z {z0}-{z1} von {shape3[0]}', flush=True)
    print(f'\nFertig: {args.out}  shape={shape3}')


if __name__ == '__main__':
    main()