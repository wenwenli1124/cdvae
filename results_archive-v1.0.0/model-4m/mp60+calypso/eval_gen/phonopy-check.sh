symprec=0.15
for i in */*.vasp; do
  spgno=`phonopy --symmetry --tolerance $symprec -c $i | head -3 |tail -1 | awk '{print $2}'`
  echo $i    $spgno
done
rm BPOSCAR PPOSCAR phonopy_symcells.yaml
