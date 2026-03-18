
import struct, os, sys
import numpy as np

# ---------- STL I/O ----------
def read_stl(path):
    """Return vertices array (N,3,3) and normals (N,3). Supports binary & ASCII STL."""
    with open(path, 'rb') as f:
        start = f.read(5); f.seek(0)
        is_ascii = start.lower().startswith(b'solid') and b'facet' in f.read(1000)
        f.seek(0)
        if not is_ascii:
            # Binary STL
            f.read(80)
            n_tri = struct.unpack('<I', f.read(4))[0]
            verts = np.empty((n_tri, 3, 3), dtype=np.float32)
            normals = np.empty((n_tri, 3), dtype=np.float32)
            for i in range(n_tri):
                data = f.read(50)  # 12 floats + 2 bytes attr
                normals[i] = struct.unpack('<fff', data[0:12])
                verts[i,0] = struct.unpack('<fff', data[12:24])
                verts[i,1] = struct.unpack('<fff', data[24:36])
                verts[i,2] = struct.unpack('<fff', data[36:48])
            return verts, normals, False
        else:
            # ASCII STL (simple parser)
            text = f.read().decode('utf-8', errors='ignore')
            tri = []; normals = []; verts_list = []
            for ln in (ln.strip() for ln in text.splitlines()):
                if ln.startswith('facet normal'):
                    parts = ln.split()
                    normals.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
                elif ln.startswith('vertex'):
                    parts = ln.split()
                    tri.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if len(tri) == 3:
                        verts_list.append(tri); tri = []
            verts = np.array(verts_list, dtype=np.float32)
            normals = np.array(normals, dtype=np.float32)
            return verts, normals, True

def write_binary_stl(path, triangles, name='part'):
    """Write triangles (M,3,3) as a valid binary STL."""
    M = triangles.shape[0]
    with open(path, 'wb') as f:
        header = (name[:80]).ljust(80, ' ').encode('ascii')
        f.write(header)
        f.write(struct.pack('<I', M))
        # Normal: recompute or set to zero; here set zero to avoid heavy math
        for tri in triangles:
            f.write(struct.pack('<fff', 0.0, 0.0, 0.0))         # normal
            for v in tri:
                f.write(struct.pack('<fff', float(v[0]), float(v[1]), float(v[2])))
            f.write(struct.pack('<H', 0))                       # attribute byte count

# ---------- Heuristic split ----------
def auto_split(verts):
    """
    Heuristic segmentation into parts using spatial percentiles.
    Returns dict: part_name -> triangle indices.
    """
    n = verts.shape[0]
    cents = verts.mean(axis=1)  # (N,3)
    x, y, z = cents[:,0], cents[:,1], cents[:,2]
    cx, cy, cz = x.mean(), y.mean(), z.mean()
    r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)

    def percentiles(a, ps): return np.percentile(a, ps)

    z_p = percentiles(z, [3,5,15,50,70,85,90,95,97])
    absx = np.abs(x-cx); absy = np.abs(y-cy)
    absx_p = percentiles(absx, [80,90,95])
    absy_p = percentiles(absy, [80,90,95])
    r_p = percentiles(r, [80,90,95])

    assigned = np.zeros(n, dtype=bool)
    parts = {}

    def pick(mask, name):
        idx = np.where(mask & (~assigned))[0]
        assigned[idx] = True
        parts[name] = idx

    # Tune these masks to your model as needed:
    pick(z <= z_p[0], 'Base')  # very low Z
    pick(z >= z_p[-1], 'Santa_Hat')  # very high Z
    pick((z >= z_p[6]) & (absx >= absx_p[1]), 'Horns')
    pick((r >= r_p[1]) & (z >= z_p[2]) & (z <= z_p[5]), 'Tail_Ornament')
    pick((absy >= absy_p[1]) & (z >= z_p[2]) & (z <= z_p[6]), 'Gift_Box_Ribbon')
    pick((z >= z_p[3]) & (z <= z_p[4]) & (r <= r_p[0]), 'Wreath')
    pick((z >= z_p[1]) & (z <= z_p[2]) & (r <= r_p[0]), 'Belly')
    pick((z >= z_p[1]) & (z <= z_p[2]) & (absx <= absx_p[0]), 'Claws')
    pick((z >= z_p[5]) & (r <= r_p[0]), 'Eyes')
    parts['Main_Body_Wings'] = np.where(~assigned)[0]
    return parts

def main(input_stl):
    if not os.path.exists(input_stl):
        print('Input STL not found:', input_stl)
        return

    print('Reading STL...')
    verts, normals, is_ascii = read_stl(input_stl)
    print('Triangles:', verts.shape[0])

    print('Splitting...')
    parts = auto_split(verts)

    created = []
    for name, idx in parts.items():
        tris = verts[idx]
        if tris.shape[0] == 0:
            print(f'Skipped {name} (no triangles detected).')
            continue
        out_name = f'{name}.stl'
        write_binary_stl(out_name, tris, name=name)
        created.append(out_name)
        print(f'Saved {out_name} | triangles: {tris.shape[0]}')

    with open('split_report.txt', 'w') as f:
        for name, idx in parts.items():
            f.write(f'{name}: {len(idx)} triangles\\n')

    print('Done. Created:', ', '.join(created))

if __name__ == '__main__':
    # Change filename if needed
    main('Festive_Dragon_Gift_big.stl')
