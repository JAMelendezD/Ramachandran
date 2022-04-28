# Ramachandran
Plots the trajectory of a selection of residues on a Ramachandran plot.

```
python ramachandran.py protein.tpr protein.xtc 10000 20000 50 1 500 result
```

Output of png images can be easily converted to an animation using Imagick convert

```
convert -delay 5 -dispose Background *.png anim.gif
```

<p align="center">
  <img width="400" src="images/many.gif">
</p>

<p align="center">
  <img width="400" src="images/many.gif">
  <img width="400" src="images/many.gif">
</p>


