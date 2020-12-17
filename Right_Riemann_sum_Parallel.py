#example to run: mpiexec -n 4 python3 Right_Riemann_sum_Parallel.py
import numpy
import sys
import math
import time
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#takes in command-line arguments [a,b,n]
#a = float(sys.argv[1])
#b = float(sys.argv[2])
#n = int(sys.argv[1])
a = float(0)
b = float(math.pi/4)
n = 50000

def f(x):
        return math.sin(x*x)
    
def integrateRange(a, b, n):
        integral = 0
        list1 = list(numpy.linspace(a,b,n))
        list2 = list1[1:int(n)]
        # n endpoints, but n trapazoids
        for x in list2:
                integral = integral + f(x)
        integral = integral* (b-a)/n
        return integral


#h is the step size. n is the total number of trapezoids
h = (b-a)/n
#local_n is the number of trapezoids each process will calculate
#note that size must divide n
local_n = n/size

#we calculate the interval that each process handles
#local_a is the starting point and local_b is the endpoint
local_a = a + rank*local_n*h
local_b = local_a + local_n*h

#initializing variables. mpi4py requires that we pass numpy objects.
integral = numpy.zeros(1)
recv_buffer = numpy.zeros(1)

# perform local computation. Each process integrates its own interval
integral[0] = integrateRange(local_a, local_b, local_n)

# communication
# root node receives results from all processes and sums them
if rank == 0:
        total = integral[0]
        for i in range(1, size):
                comm.Recv(recv_buffer, ANY_SOURCE)
                total += recv_buffer[0]
else:
        # all other process send their result
        # dest - main process = 0
        comm.Send(integral,dest = 0)

# root process prints results
if comm.rank == 0:
        print ("With n =", n, "trapezoids, our estimate of the integral from"\
        , a, "to", b, "is", total,"/n time: ",time.clock())

