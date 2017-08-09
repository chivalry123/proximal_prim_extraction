
from __future__ import print_function, division

import os
import re
import sys
import numpy as np
from scipy.spatial import ConvexHull
# import glob
# import itertools
# from numpy.linalg import inv
import argparse
import scipy
import operator

from numpy import all, array, uint8
from math import log, floor

from cvxopt import normal, matrix

from cvxopt.glpk import ilp

__author__ = "Wenxuan Huang"
__email__ = "key01027@mit.edu"
__date__ = "2016-09-02"
__version__ = "0.1"

# Greatest common divisor of more than 2 numbers.  Am I terrible for doing it this way?

def gcd(*numbers):
    """Return the greatest common divisor of the given integers"""
    from fractions import gcd
    return reduce(gcd, numbers)

# Least common multiple is not in standard libraries? It's in gmpy, but this is simple enough:

def lcm(*numbers):
    """Return lowest common multiple."""
    def lcm(a, b):
        return (a * b) // gcd(a, b)
    return reduce(lcm, numbers, 1)

# Assuming numbers are positive integers...


def read_PRIM(prim_file):
    with open(prim_file, 'r') as fp:
        fp.readline()
        a = float(fp.readline())
        avec = np.zeros((3, 3))
        avec[0] = (map(float, fp.readline().split()))
        avec[1] = map(float, fp.readline().split())
        avec[2] = map(float, fp.readline().split())
        avec *= a
        nsites = map(int, fp.readline().split())
        cooformat = fp.readline()
        if cooformat.strip() not in ["Direct", "direct"]:
            print("Error: This script can only handle direct coordinates!")
            print(cooformat)
            sys.exit()
        coords = []
        species = []
        for i in range(np.sum(nsites)):
            line = fp.readline()
            coords.append(map(float, line.split()[:3]))
            # FIXME: make sure that all coordinates are in [0:1]
            species.append(line.split()[3:])
        coords = np.array(coords)
    return (avec, coords, species)

def read_POSCAR(prim_file):
    with open(prim_file, 'r') as fp:
        fp.readline()
        a = float(fp.readline())
        avec = np.zeros((3, 3))
        avec[0] = (map(float, fp.readline().split()))
        avec[1] = map(float, fp.readline().split())
        avec[2] = map(float, fp.readline().split())
        avec *= a
        fp.readline()
        nsites = map(int, fp.readline().split())
        cooformat = fp.readline()
        if cooformat.strip() not in ["Direct", "direct"]:
            print("Error: This script can only handle direct coordinates!")
            print(cooformat)
            sys.exit()
        coords = []
        species = []
        for i in range(np.sum(nsites)):
            line = fp.readline()
            coords.append(map(float, line.split()[:3]))
            # FIXME: make sure that all coordinates are in [0:1]
            species.append(line.split()[3:])
        coords = np.array(coords)
    return (avec, coords, species)



def get_CE_sites(coords, species):
    sites = []
    species_index = []
    for i in range(len(coords)):
        if species[i] > 1:
            sites.append(i)
            if species[i] not in species_index:
                species_index.append(species[i])
    return (sites, species_index)

def positive_modulo(i,n):
    return (i % n + n) % n;

def floor_int_division(up, down):
    float_tmp=float(up)/down+1e-6;
    return int(np.floor(float_tmp));



def convert_out_of_cell_index_to_incell_index(out_of_cell_index,periodicity):
    a0=int(round(periodicity[0]))
    a1=int(round(periodicity[1]))
    a2=int(round(periodicity[2]))
    a3=int(round(periodicity[3]))
    a4=int(round(periodicity[4]))
    a5=int(round(periodicity[5]))
    x_0=int(round(out_of_cell_index[0]))
    y_0=int(round(out_of_cell_index[1]))
    z_0=int(round(out_of_cell_index[2]))
    p_0=int(round(out_of_cell_index[3]))

    x_1=((x_0)-floor_int_division(y_0-floor_int_division(z_0,a5)*a4,a2)*a1-floor_int_division(z_0,a5)*a3)% a0
    y_1= (((y_0)-floor_int_division(z_0,a5)*a4)%a2)
    z_1= ((z_0) % a5)
    p_1=p_0

    return  (x_1,y_1,z_1,p_1)



class spin_configuration_class(object):
    ## x,y,z,p all starts from 0
    def __init__(self, configuration,periodicity):
        self.element_count=None
        self.inverse_configuration={}
        self.configuration=configuration
        self.periodicity=tuple(map(lambda x : int(round(x)) ,periodicity))
        self.subcell_number=self.calculate_subscell_number()
        self.concentration=self.calculate_concentration()

    def expand_configuration(self):
        pass

    def evaluate_at_index(self,index):
        if index in self.configuration:
            return self.configuration[index]
        else:
            reduced_index=convert_out_of_cell_index_to_incell_index(index,self.periodicity)
            self.configuration[index]=self.configuration[reduced_index]
            return self.configuration[index]

    def calculate_subscell_number(self):
        subcell_number=0
        for i in range(1000):
            if (0,0,0,i) in self.configuration:
                subcell_number+=1
            else:
                break
        return subcell_number

    def calculate_concentration(self):
        concentration_dict={}
        for i in range(self.periodicity[0]):
            for j in range(self.periodicity[2]):
                for k in range(self.periodicity[5]):
                    for p in range(self.subcell_number):
                        specie_now=self.configuration[(i,j,k,p)]
                        if specie_now in concentration_dict:
                            concentration_dict[specie_now]+=1
                        else:
                            concentration_dict[specie_now]=1

                        if specie_now in self.inverse_configuration:
                            self.inverse_configuration[specie_now].append((i,j,k,p))
                        else:
                            self.inverse_configuration[specie_now]=[]
                            self.inverse_configuration[specie_now].append((i,j,k,p))



        self.element_count=concentration_dict.copy()
        total_atoms=sum(concentration_dict.values())
        print("total atoms ",total_atoms)
        for key in concentration_dict:
            concentration_dict[key]/=float(total_atoms)

        return concentration_dict






def calculate_error_rate_for_this_periodicity_vector(periodicity_vector_now,spin_configuration_poscar_class):

    a0=spin_configuration_poscar_class.periodicity[0]
    a2=spin_configuration_poscar_class.periodicity[2]
    a5=spin_configuration_poscar_class.periodicity[5]

    total_count=0
    error_count=0

    for i in range(a0):
        for j in range(a2):
            for k in range(a5):
                for p in range(spin_configuration_poscar_class.subcell_number):
                    original_site=(i,j,k,p)
                    i0=periodicity_vector_now[0]+i
                    j0=periodicity_vector_now[1]+j
                    k0=periodicity_vector_now[2]+k
                    to_compare_site=(i0,j0,k0,p)
                    total_count+=1
                    if ( spin_configuration_poscar_class.evaluate_at_index(original_site) \
                        !=spin_configuration_poscar_class.evaluate_at_index(to_compare_site)):
                        error_count+=1

    error_rate=float(error_count)/total_count
    return error_rate


def classify_vector(periodicity_vector):
    if periodicity_vector[2]==0 and periodicity_vector[1]==0:
        return 'x'
    elif periodicity_vector[2]==0 and periodicity_vector[1]!=0:
        return 'y'
    if periodicity_vector[2]!=0:
        return 'z'

    assert False

def obtain_common_periodicity(target_periodicity,based_periodicity):
    #the standard terms here is to compute the join of hermite normal form, here we would just use some very inefficient approach, but should be fine
    print("target_periodicity")
    print(target_periodicity)
    print("base_periodicity")
    print(based_periodicity)
    target_x=np.array([target_periodicity[0],0,0])
    target_y=np.array([target_periodicity[1],target_periodicity[2],0])
    target_z=np.array([target_periodicity[3],target_periodicity[4],target_periodicity[5]])

    a2_a4_lcm=lcm(target_y[1],target_z[1])
    if target_z[1]!=0:
        target_z=target_z*a2_a4_lcm/target_z[1]-target_y*a2_a4_lcm/target_y[1]
        if target_z[0]!=0:
            a3_a0_lcm=lcm(target_z[0],target_x[0])
            target_z=target_z*a3_a0_lcm/target_z[0]-target_x*a3_a0_lcm/target_x[0]
            if target_y[0]!=0:
                a1_a0_lcm=lcm(target_y[0],target_x[0])
                target_y=target_y*a1_a0_lcm/target_y[0]-target_x*a1_a0_lcm/target_x[0]

    print("target_x_y_z is", target_x,target_y,target_z)

    common_periodicity=(lcm(target_x[0],based_periodicity[0]),0,lcm(target_y[1],based_periodicity[2]),0,0,lcm(target_z[2]\
        ,based_periodicity[5]))

    common_periodicity=tuple(map(lambda x: int(round(x)),common_periodicity))

    return common_periodicity


def construct_configuration_based_on_target_periodicty(target_periodicity,based_configuration_class):
    based_periodicity=based_configuration_class.periodicity
    assert abs(based_periodicity[1])+abs(based_periodicity[3])+abs(based_periodicity[4])==0
    common_periodicity=obtain_common_periodicity(target_periodicity,based_periodicity)


    # print("common_periodicity[0]",repr( common_periodicity[0]))
    # print("target_periodicity[0]",repr( target_periodicity[0]))
    xs_to_check=common_periodicity[0]//target_periodicity[0]
    # print("xs_to_check",xs_to_check)
    ys_to_check=common_periodicity[2]//target_periodicity[2]
    zs_to_check=common_periodicity[5]//target_periodicity[5]
    # print("zs to check ", zs_to_check)

    target_x=np.array([target_periodicity[0],0,0])
    target_y=np.array([target_periodicity[1],target_periodicity[2],0])
    target_z=np.array([target_periodicity[3],target_periodicity[4],target_periodicity[5]])


    constructed_spin_configuration={}

    for i in range(target_periodicity[0]):
        for j in range(target_periodicity[2]):
            for k in range(target_periodicity[5]):
                for p in range(based_configuration_class.subcell_number):
                    original_site=np.array([i,j,k])
                    original_index=tuple( np.concatenate((original_site,np.array([p]))))
                    possible_species={}
                    for n1 in range(xs_to_check):
                        for n2 in range(ys_to_check):
                            for n3 in range(zs_to_check):
                                target_site=original_site+n1*target_x+n2*target_y+n3*target_z
                                # print("target site is ",target_site)
                                target_index=tuple( np.concatenate((target_site,np.array([p]))))
                                # print("target index is ",target_index)
                                specie_now=based_configuration_class.evaluate_at_index(target_index)
                                # print("specie_now is ",specie_now)
                                if specie_now not in possible_species:
                                    possible_species[specie_now]=1
                                else:
                                    possible_species[specie_now]+=1
                    # print("possible_species are")
                    # print(possible_species)
                    max_element=max(possible_species.iteritems(), key=operator.itemgetter(1))[0]
                    # print("max_element is",max_element)
                    constructed_spin_configuration[original_index]=max_element
                    # possible_species_debug={}
                    # for n1 in range(xs_to_check*5):
                    #     for n2 in range(ys_to_check*5):
                    #         for n3 in range(zs_to_check*5):
                    #             target_site=original_site+n1*target_x+n2*target_y+n3*target_z
                    #             # print("target site is ",target_site)
                    #             target_index=tuple( np.concatenate((target_site,np.array([p]))))
                    #             # print("target index is ",target_index)
                    #             specie_now=based_configuration_class.evaluate_at_index(target_index)
                    #             # print("specie_now is ",specie_now)
                    #             if specie_now not in possible_species_debug:
                    #                 possible_species_debug[specie_now]=1
                    #             else:
                    #                 possible_species_debug[specie_now]+=1
                    # print("possible_species_debug are")
                    # print(possible_species_debug)

    constructed_spin_configuration_class=spin_configuration_class(constructed_spin_configuration,target_periodicity)
    return  constructed_spin_configuration_class

    ## need to do some mathemtical checking, done

def print_to_poscar(spin_configuration_class,avec_prim,coords_prim,species_prim,destination):

    file_content=""
    target_primitive_x=avec_prim[0]*spin_configuration_class.periodicity[0]
    target_primitive_y=avec_prim[0]*spin_configuration_class.periodicity[1]+\
        avec_prim[1]*spin_configuration_class.periodicity[2]
    target_primitive_z=avec_prim[0]*spin_configuration_class.periodicity[3]+\
        avec_prim[1]*spin_configuration_class.periodicity[4]+\
        avec_prim[2]*spin_configuration_class.periodicity[5]


    a0=spin_configuration_class.periodicity[0]
    a1=spin_configuration_class.periodicity[1]
    a2=spin_configuration_class.periodicity[2]
    a3=spin_configuration_class.periodicity[3]
    a4=spin_configuration_class.periodicity[4]
    a5=spin_configuration_class.periodicity[5]


    file_content+="generated from extract small configs from GMC\n"
    file_content+="1.0\n"
    file_content+="{:.8f}{:17.8f}{:17.8f}\n".format(target_primitive_x[0],target_primitive_x[1],target_primitive_x[2])
    file_content+="{:.8f}{:17.8f}{:17.8f}\n".format(target_primitive_y[0],target_primitive_y[1],target_primitive_y[2])
    file_content+="{:.8f}{:17.8f}{:17.8f}\n".format(target_primitive_z[0],target_primitive_z[1],target_primitive_z[2])
    # file_content+=spin_configuration_class.element_count
    file_content+='  '.join(spin_configuration_class.inverse_configuration.keys())+"\n"
    file_content+='  '.join([str(len(x)) for x in \
        spin_configuration_class.inverse_configuration.values()])+"\n"
    file_content+='direct\n'

    # print(" spin_configuration_class.inverse_configuration is")
    # print( spin_configuration_class.inverse_configuration)
    for (keys,values) in  spin_configuration_class.inverse_configuration.iteritems():
        # print("keys is ",keys)
        # print("value is ",values)
        coord_list_now=[]
        for v in values:
            # print("v is ",v)
            x=v[0]+coords_prim[v[3]][0]
            y=v[1]+coords_prim[v[3]][1]
            z=v[2]+coords_prim[v[3]][2]
            x_transformed=x/a0-y/a2*a1/a0+z*a1*a4/a0/a2/a5-z/a5*a3/a0
            y_transformed=y/a2-z/a5*a4/a2
            z_transformed=z/a5
            x_transformed=x_transformed-floor(x_transformed)
            y_transformed=y_transformed-floor(y_transformed)
            z_transformed=z_transformed-floor(z_transformed)
            coord_list_now.append((x_transformed,y_transformed,z_transformed))
        coord_list_now.sort()
        for v in coord_list_now:
            x_transformed=v[0]
            y_transformed=v[1]
            z_transformed=v[2]

            file_content+="{:.8f}{:17.8f}{:17.8f}   ".format(x_transformed,y_transformed,z_transformed)+keys+"\n"

    # print(file_content)



    f=open(destination,'w')
    f.write(file_content)
    f.close()

    pass


#only support orthogonal supercell now


if (__name__ == "__main__"):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--PRIM", "-pr",
        help="Path to a CASM PRIM file (default: PRIM).",
        default="PRIM")

    parser.add_argument(
        "--poscar", "-po",
        help="Path to struct_xyz file.",
        default="")

    parser.add_argument(
        "--MAXSIZE", "-size",
        help="Maxsize for approximated cell",
        type=float,
        default=None)

    parser.add_argument(
        "--WriteToPoscar","-wp",
        help="Path to write output poscar .",
        default=None
    )


    args = parser.parse_args()
    maxsize=float(args.MAXSIZE)+0.5
    output_poscar_destination=args.WriteToPoscar


    (avec_prim, coords_prim, species_prim)=read_PRIM(args.PRIM)
    # print("(avec_prim, coords_prim, species_prim)")
    # print((avec_prim, coords_prim, species_prim))

    (sites_prim, species_index_prim) = get_CE_sites(coords_prim, species_prim)

    # print("(sites_prim, species_index_prim)")
    # print((sites_prim, species_index_prim))

    (avec_poscar, coords_poscar, species_poscar)=read_POSCAR(args.poscar)
    # print("(avec_poscar, coords_poscar, species_poscar)")
    # print((avec_poscar, coords_poscar, species_poscar))

    supercell_matrix_from_prim_to_poscar=np.linalg.solve(avec_prim.T,avec_poscar.T)
    # print("supercell_matrix_from_prim_to_poscar")
    # print(supercell_matrix_from_prim_to_poscar)

    rounded_supercell_matrix=(np.round(supercell_matrix_from_prim_to_poscar))

    rounded_supercell_matrix.astype(int)
    # print(rounded_supercell_matrix)

    residual=supercell_matrix_from_prim_to_poscar-rounded_supercell_matrix
    residual_norm=scipy.linalg.norm(residual,1)


    # print(residual)
    # print(scipy.linalg.norm(residual,1))
    assert residual_norm<1e-4

    configurations_poscar={}

    for i in range (len( coords_poscar)):
        coords_poscar_cartesian_now=avec_poscar.T.dot( coords_poscar[i])
        # print(coords_poscar_cartesian_now)

        freactional_coords_according_to_PRIM=np.linalg.solve(avec_prim.T,coords_poscar_cartesian_now)
        # print("freactional_coords_according_to_PRIM")
        # print(freactional_coords_according_to_PRIM)

        supercell_index=np.floor( freactional_coords_according_to_PRIM+1e-4)
        # print("supercell_index")
        # print(supercell_index)

        subcell_position= freactional_coords_according_to_PRIM-supercell_index
        # print("subcell_position")
        # print(subcell_position)

        residual_position=np.abs(coords_prim-subcell_position)
        # print("residual_position")
        # print(residual_position)

        sum_error=np.sum(residual_position,1)


        subcell_index=np.argmin(sum_error)
        assert np.amin(sum_error)<1e-4

        total_index=np.hstack((supercell_index,subcell_index))
        total_index=tuple(total_index)
        # print("total_index")
        # print(total_index)

        configurations_poscar[total_index]=species_poscar[i][0]

    # print("configurations_poscar is ",configurations_poscar)
    a0=int (rounded_supercell_matrix[0,0])
    a1=0
    a2=int( rounded_supercell_matrix[1,1])
    a3=0
    a4=0
    a5=int(rounded_supercell_matrix[2,2])
    periodicity_poscar=(a0,a1,a2,a3,a4,a5)
    assert abs(sum(sum(abs(rounded_supercell_matrix))))-a0-a2-a5==0
    #fixme: currently only work on tetragonal super cell, need to transform to hermite normal if using other supercell
    print("a0,a2,a5")
    print(a0,a2,a5)


    spin_configuration_poscar_class=spin_configuration_class(configurations_poscar,periodicity_poscar)


    periodicity_vector_list=[]
    for periodicity_x in range(0,a0+1):
        if not periodicity_x==0:
            periodicity_vector_list.append((periodicity_x,0,0))

        for periodicity_y in range(0,a2+1):
            if not periodicity_y==0 and periodicity_x!=a0:
                periodicity_vector_list.append((periodicity_x,periodicity_y,0))

            for periodicity_z in range(1,a5+1):
                if periodicity_z!=0 and periodicity_y!=a2 and periodicity_x!=a0:
                    periodicity_vector_list.append((periodicity_x,periodicity_y,periodicity_z))

    periodicity_vector_list.sort()
    # print(periodicity_list)

    error_rate_list=[]
    for i in range(len(periodicity_vector_list)):
        periodicity_vector_now=periodicity_vector_list[i]
        error_rate_now=calculate_error_rate_for_this_periodicity_vector(periodicity_vector_now,spin_configuration_poscar_class)
        error_rate_list.append(error_rate_now)


    zip_list_periodicity_error=zip(periodicity_vector_list,error_rate_list)
    zip_list_periodicity_error.sort( key = lambda t:t[1] )
    # print("zip_list_periodicity_error is")
    # print(zip_list_periodicity_error)

    zip_list_periodicity_error_x=[]
    zip_list_periodicity_error_y=[]
    zip_list_periodicity_error_z=[]
    for i in range(len(zip_list_periodicity_error)):
        type_now=classify_vector(zip_list_periodicity_error[i][0])
        if type_now=='x':
            zip_list_periodicity_error_x.append(zip_list_periodicity_error[i])
        elif type_now=='y':
            zip_list_periodicity_error_y.append(zip_list_periodicity_error[i])
        elif type_now=='z':
            zip_list_periodicity_error_z.append(zip_list_periodicity_error[i])

    print("zip_list_periodicity_error_x")
    print(zip_list_periodicity_error_x)
    print("zip_list_periodicity_error_y")
    print(zip_list_periodicity_error_y)
    print("zip_list_periodicity_error_z")
    print(zip_list_periodicity_error_z)




    x_var_size=len(zip_list_periodicity_error_x)
    y_var_size=len(zip_list_periodicity_error_y)
    z_var_size=len(zip_list_periodicity_error_z)

    error_vector_x=np.array( [i[1] for i in zip_list_periodicity_error_x])
    error_vector_y=np.array( [i[1] for i in zip_list_periodicity_error_y])
    error_vector_z=np.array( [i[1] for i in zip_list_periodicity_error_z])

    x_log_size_measure_vector=np.array([log(i[0][0]) for i in zip_list_periodicity_error_x])
    y_log_size_measure_vector=np.array([log(i[0][1]) for i in zip_list_periodicity_error_y])
    z_log_size_measure_vector=np.array([log(i[0][2]) for i in zip_list_periodicity_error_z])



    # (status, x) = ilp(c, G, h, A, b, I, B)
    #         minimize    c'*x
    #     subject to  G*x <= h
    #                 A*x = b
    #                 x[k] is integer for k in I
    #                 x[k] is binary for k in B

    c_error_part=np.concatenate((error_vector_x,error_vector_y,error_vector_z))
    c_size_part=1e-6*np.concatenate((x_log_size_measure_vector,y_log_size_measure_vector,z_log_size_measure_vector))
    c_total=c_error_part+c_size_part
    # print("c_error_part")
    # print(c_error_part)

    A_new_line=np.concatenate((np.ones(x_var_size),np.zeros(y_var_size),np.zeros(z_var_size)))
    # print("A_new_line")
    # print(A_new_line)

    A_matrix=A_new_line[:]
    A_new_line=np.concatenate((np.zeros(x_var_size),np.ones(y_var_size),np.zeros(z_var_size)))
    A_matrix=np.vstack((A_matrix,A_new_line))

    A_new_line=np.concatenate((np.zeros(x_var_size),np.zeros(y_var_size),np.ones(z_var_size)))
    A_matrix=np.vstack((A_matrix,A_new_line))

    b_vector=np.ones(3)

    # print("A_matrix")
    # print(A_matrix)

    G_matrix=np.hstack((x_log_size_measure_vector,y_log_size_measure_vector,z_log_size_measure_vector))
    G_matrix.shape=(1,G_matrix.shape[0])
    # print("G_matrix")
    # print(G_matrix)
    h_matrix=np.array([log(maxsize)])

    B_set=range(x_var_size+y_var_size+z_var_size)


    # output_cvxopt = ilp(c=matrix(c_total), G=matrix(G_matrix), h=matrix(h_matrix),A=matrix(A_matrix), b=matrix(b_vector))
    output_cvxopt = ilp(c=matrix(c_total), G=matrix(G_matrix), h=matrix(h_matrix),A=matrix(A_matrix), b=matrix(b_vector),I=set(B_set), B=set(B_set))

    # print("solution np.array(output_cvxopt[1])")
    # print(np.array(output_cvxopt[1]))
    solutions=np.array(output_cvxopt[1])
    x_solution=solutions[0:x_var_size].ravel()
    y_solution=solutions[x_var_size:x_var_size+y_var_size].ravel()
    z_solution=solutions[x_var_size+y_var_size:x_var_size+y_var_size+z_var_size].ravel()


    x_periodicity_index=np.nonzero(x_solution)[0][0]
    y_periodicity_index=np.nonzero(y_solution)[0][0]
    z_periodicity_index=np.nonzero(z_solution)[0][0]
    x_periodicity=zip_list_periodicity_error_x[x_periodicity_index][0]
    y_periodicity=zip_list_periodicity_error_y[y_periodicity_index][0]
    z_periodicity=zip_list_periodicity_error_z[z_periodicity_index][0]
    total_error_approximated=zip_list_periodicity_error_x[x_periodicity_index][1]+zip_list_periodicity_error_y[y_periodicity_index][1]\
        +zip_list_periodicity_error_z[z_periodicity_index][1]

    soln_system=(x_periodicity,y_periodicity,z_periodicity,total_error_approximated)
    print("soln_system is")
    print(soln_system)

    target_periodicity=(x_periodicity[0],y_periodicity[0],y_periodicity[1],z_periodicity[0],z_periodicity[1],z_periodicity[2])


    constructed_spin_configuration_class=construct_configuration_based_on_target_periodicty(target_periodicity,spin_configuration_poscar_class)
    # print(spin_configuration_poscar_class.concentration)
    print("input concentration")
    print(spin_configuration_poscar_class.concentration)
    print("output concentration")
    print(constructed_spin_configuration_class.concentration)

    print_to_poscar(constructed_spin_configuration_class,avec_prim,coords_prim,species_prim,output_poscar_destination)

