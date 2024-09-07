import os
import subprocess
import warnings
import shutil
from itertools import product
from pathlib import Path

import click
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Potcar
from pymatgen.io.vasp.sets import MPRelaxSet
from joblib import Parallel, delayed
from tqdm import tqdm


def potcar2distmat(potcar: Potcar):
    rc = [p.RCORE for p in potcar]
    distmat = np.asarray([(i + j) * 0.529177 for i, j in product(rc, repeat=2)])
    distmat = distmat.reshape(len(rc), len(rc))
    return distmat


def get_conversion_factor(spg: int):
    factors = {
        1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 1, 7: 1, 8: 2, 9: 2, 10: 1,
        11: 1, 12: 2, 13: 1, 14: 1, 15: 2, 16: 1, 17: 1, 18: 1, 19: 1, 20: 2,
        21: 2, 22: 4, 23: 2, 24: 2, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1,
        31: 1, 32: 1, 33: 1, 34: 1, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2, 40: 2,
        41: 2, 42: 4, 43: 4, 44: 2, 45: 2, 46: 2, 47: 1, 48: 1, 49: 1, 50: 1,
        51: 1, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 1,
        61: 1, 62: 1, 63: 2, 64: 2, 65: 2, 66: 2, 67: 2, 68: 2, 69: 4, 70: 4,
        71: 2, 72: 2, 73: 2, 74: 2, 75: 1, 76: 1, 77: 1, 78: 1, 79: 2, 80: 2,
        81: 1, 82: 2, 83: 1, 84: 1, 85: 1, 86: 1, 87: 2, 88: 2, 89: 1, 90: 1,
        91: 1, 92: 1, 93: 1, 94: 1, 95: 1, 96: 1, 97: 2, 98: 2, 99: 1, 100: 1,
        101: 1, 102: 1, 103: 1, 104: 1, 105: 1, 106: 1, 107: 2, 108: 2, 109: 2, 110: 2,
        111: 1, 112: 1, 113: 1, 114: 1, 115: 1, 116: 1, 117: 1, 118: 1, 119: 2, 120: 2,
        121: 2, 122: 2, 123: 1, 124: 1, 125: 1, 126: 1, 127: 1, 128: 1, 129: 1, 130: 1,
        131: 1, 132: 1, 133: 1, 134: 1, 135: 1, 136: 1, 137: 1, 138: 1, 139: 2, 140: 2,
        141: 2, 142: 2, 143: 1, 144: 1, 145: 1, 146: 3, 147: 1, 148: 3, 149: 1, 150: 1,
        151: 1, 152: 1, 153: 1, 154: 1, 155: 3, 156: 1, 157: 1, 158: 1, 159: 1, 160: 3,
        161: 3, 162: 1, 163: 1, 164: 1, 165: 1, 166: 3, 167: 3, 168: 1, 169: 1, 170: 1,
        171: 1, 172: 1, 173: 1, 174: 1, 175: 1, 176: 1, 177: 1, 178: 1, 179: 1, 180: 1,
        181: 1, 182: 1, 183: 1, 184: 1, 185: 1, 186: 1, 187: 1, 188: 1, 189: 1, 190: 1,
        191: 1, 192: 1, 193: 1, 194: 1, 195: 1, 196: 4, 197: 2, 198: 1, 199: 2, 200: 1,
        201: 1, 202: 4, 203: 4, 204: 2, 205: 1, 206: 2, 207: 1, 208: 1, 209: 4, 210: 4,
        211: 2, 212: 1, 213: 1, 214: 2, 215: 1, 216: 4, 217: 2, 218: 1, 219: 4, 220: 2,
        221: 1, 222: 1, 223: 1, 224: 1, 225: 4, 226: 4, 227: 4, 228: 4, 229: 2, 230: 2
    }
    return factors[spg]



def vasp2inputdat(poscar, potcar, dist_ratio=0.7, popsize=1, spg=None, unitcell=None):
    distmat = potcar2distmat(potcar) * dist_ratio
    ds = ""
    for line in distmat:
        ds += " ".join(map(str, line)) + "\n"

    natoms = poscar.natoms
    if spg is not None:
        spg_line = f"SpeSpaceGroup = {spg} {spg}\n"
        if unitcell is None:
            raise ValueError("templates are unicell or not must be given when spg is specified")
        if unitcell:
            factor = get_conversion_factor(spg)
            natoms = list(map(lambda x: x // factor, natoms))

    inputdat = (
        f"SystemName = {''.join(poscar.site_symbols)}\n"
        f"NumberOfSpecies = {len(poscar.site_symbols)}\n"
        f"NameOfAtoms = {' '.join(poscar.site_symbols)}\n"
        f"NumberOfAtoms = {' '.join(map(str, natoms))}\n"
        "NumberOfFormula = 1 1\n"
        f"Volume = {poscar.structure.volume}\n"
        "@DistanceOfIon\n"
        f"{ds}"
        "@End\n"
        "Ialgo = 2\n"
        "PsoRatio = 0.6\n"
        f"PopSize = {popsize}\n"
        "ICode = 15\n"
        "NumberOfLbest = 4\n"
        "NumberOfLocalOptim = 3\n"
        "Command = sh submit.sh\n"
        "MaxStep = 5\n"
        "PickUp = F\n"
        "PickStep = 5\n"
        "Parallel = F\n"
        "Split = T\n"
    )
    if spg is not None:
        inputdat += spg_line
    return inputdat


def prepare_calypso_one(
    indir, fposcar, dist_ratio, popsize, calypsocmd, timeout, spg, unitcell, label=""
):
    if len(label) > 0:
        outdir = indir.with_name(f"{indir.name}.calypso-{label}")
    else:
        outdir = indir.with_name(f"{indir.name}.calypso")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poscar = Poscar.from_file(fposcar)
        potcar = Potcar.from_file(fposcar.with_name("POTCAR"))
    calypsodir = outdir.joinpath(fposcar.parent.relative_to(indir))
    calypsodir.mkdir(parents=True, exist_ok=True)
    with open(calypsodir.joinpath("input.dat"), 'w') as f:
        f.write(vasp2inputdat(poscar, potcar, dist_ratio, popsize, spg, unitcell))
    for f in ["INCAR", "POTCAR", "KPOINTS"]:
        try:
            shutil.copy(fposcar.with_name(f), calypsodir)
        except FileNotFoundError:
            pass
            # logger.warning(f"{str(fposcar.with_name(f))} not found")
    # run calypso
    try:
        os.remove(calypsodir.joinpath("step"))
    except FileNotFoundError:
        pass
    with open(calypsodir.joinpath("caly.log"), "w") as calylog:
        try:
            proc = subprocess.run(
                calypsocmd,
                stdout=calylog,
                stderr=subprocess.STDOUT,
                cwd=calypsodir,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            pass
        else:
            # split POSCAR_* to subdir
            if proc.returncode != 0:
                print(f"Calling {calypsocmd} failed in {calypsodir}")
            else:
                for popi in range(1, popsize + 1):
                    calcdir = calypsodir.joinpath(f"calc/{popi}")
                    calcdir.mkdir(parents=True, exist_ok=True)
                    shutil.move(calypsodir / f"POSCAR_{popi}", calcdir / "POSCAR")
                    for f in ["INCAR", "POTCAR", "KPOINTS"]:
                        try:
                            shutil.copy(calypsodir / f, calcdir)
                        except FileNotFoundError:
                            pass
        finally:
            # clean dir
            for pyfile in calypsodir.glob("*.py"):
                os.remove(pyfile)


@click.command
@click.argument("indir", nargs=-1)
@click.option("-j", "--njobs", type=int, default=1, help="default 1")
@click.option("-r", "--dist_ratio", type=float, default=0.7, help="distance ratio on DistanceOfIon (default 0.7)")
@click.option("-p", "--popsize", type=int, default=1, help="Popsize (default 1)")
@click.option("-c", "--calypsocmd", type=str, default="calypso.x", help="Popsize (default 1)")
@click.option("-t", "--timeout", type=float, default=180, help="Each timeout seconds (default 180s)")
@click.option("--spg", type=int, default=None, help="Specify space group, default None for not set")
@click.option("--unitcell", type=bool, default=None, help="Input templates are unit cell, must be given when `spg` is specified")
@click.option("--label", default="", help="label")
def prepare_calypso(
    njobs, indir, dist_ratio, popsize, calypsocmd, timeout, spg, unitcell, label
):
    for d in indir:
        d = Path(d)
        fposcarlist = list(d.rglob("POSCAR"))
        Parallel(njobs, backend="multiprocessing")(
            delayed(prepare_calypso_one)(
                d, fposcar, dist_ratio, popsize, calypsocmd, timeout, spg, unitcell, label
            )
            for fposcar in tqdm(fposcarlist, ncols=79)
        )


if __name__ == "__main__":
    prepare_calypso()
