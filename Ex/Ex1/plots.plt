set term postscript color
set output 'tunnel.eps'

set title 'Tunneling Effect'
set key top left
set xlabel 'E'
set ylabel 'T'

plot 'tunnel.dat' notitle

#--------------------------------

set term postscript color
set output 'delta.eps'

set title 'Timestep stability'
set key top left
set xlabel 'delta_x'
set ylabel 'coeff'
set logscale x

plot 'delta.dat' using 1:2 t 'A', \
     'delta.dat' using 1:3 t 'B'

#--------------------------------

set term postscript color
set output 'Tdep.eps'

set title 'T dependency on a'
set key top left
set xlabel 'a'
set ylabel 'T'
unset logscale
set logscale y

plot 'Tdep.dat' notitle
