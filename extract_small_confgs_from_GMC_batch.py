
from __future__ import print_function, division

import pickle
import errno
import os
import re
import sys
import numpy as np
# from scipy.spatial import ConvexHull
# import glob
# import itertools
# from numpy.linalg import inv
import argparse
import scipy
from scipy import linalg
import operator
import random
import pymatgen
import pymatgen.analysis
from pymatgen.analysis.structure_matcher import StructureMatcher
from pprint import pprint

# import pymatgen.analysis.structure_matcher as structure_matcher
# from pymatgen import analysis
# from pymatgen.analysis.structure_matcher import StructureMatcher
# import pymatgen.analysis.structure_matcher as StructureMatcher

from numpy import all, array, uint8
from math import log, floor

from cvxopt import normal, matrix

from cvxopt.glpk import ilp

__author__ = "Wenxuan Huang"
__email__ = "key01027@mit.edu"
__date__ = "2016-09-02"
__version__ = "0.1"

# Greatest common divisor of more than 2 numbers.  Am I terrible for doing it this way?

def save_obj(obj, name ):
    dir_name = "obj/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

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
    def __init__(self, configuration,periodicity,species_prim):
        self.element_count=None
        self.inverse_configuration={}
        self.configuration=configuration
        self.periodicity=tuple(map(lambda x : int(round(x)) ,periodicity))
        self.subcell_number=self.calculate_subscell_number(species_prim)
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

    def calculate_subscell_number(self,species_prim):
        return len(species_prim)
        # subcell_number=0
        # for i in range(1000):
        #     if (0,0,0,i) in self.configuration:
        #         subcell_number+=1
        #     else:
        #         break
        # return subcell_number

    def calculate_concentration(self):
        concentration_dict={}
        self.inverse_configuration={}

        for i in range(self.periodicity[0]):
            for j in range(self.periodicity[2]):
                for k in range(self.periodicity[5]):
                    for p in range(self.subcell_number):
                        if (i,j,k,p) not in self.configuration:
                            self.configuration[(i,j,k,p)]="Vac"
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
        # print("total atoms ",total_atoms)
        for key in concentration_dict:
            concentration_dict[key]/=float(total_atoms)

        return concentration_dict






def calculate_error_rate_for_this_periodicity_vector(periodicity_vector_now,spin_configuration_poscar_class):

    a0=spin_configuration_poscar_class.periodicity[0]
    a2=spin_configuration_poscar_class.periodicity[2]
    a5=spin_configuration_poscar_class.periodicity[5]

    total_count=0
    error_count=0

    # print ("a0,a2,a5,spin_configuration_poscar_class.subcell_number")
    # print (a0,a2,a5,spin_configuration_poscar_class.subcell_number)
    for i in range(a0):
        # print ("calculating error for i = ",i)
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

    # print("target_x_y_z is", target_x,target_y,target_z)


    a2_a4_lcm=lcm(target_y[1],target_z[1])
    if target_z[1]!=0:
        # print(target_z*a2_a4_lcm/target_z[1])
        # print(target_y*a2_a4_lcm/target_y[1])
        target_z=target_z*a2_a4_lcm/target_z[1]-target_y*a2_a4_lcm/target_y[1]
        # print("target_x_y_z is", target_x,target_y,target_z)
        target_z[0]=target_z[0]%target_x[0]

        if target_z[0]!=0:
            a3_a0_lcm=lcm(target_z[0],target_x[0])
            target_z=target_z*a3_a0_lcm/target_z[0]-target_x*a3_a0_lcm/target_x[0]
            # print("target_x_y_z is", target_x,target_y,target_z)
            target_y[0]=target_y[0]%target_x[0]

            if target_y[0]!=0:
                a1_a0_lcm=lcm(target_y[0],target_x[0])
                target_y=target_y*a1_a0_lcm/target_y[0]-target_x*a1_a0_lcm/target_x[0]
                # print("target_x_y_z is", target_x,target_y,target_z)


    # print("target_x_y_z is", target_x,target_y,target_z)

    common_periodicity=(lcm(target_x[0],based_periodicity[0]),0,lcm(target_y[1],based_periodicity[2]),0,0,lcm(target_z[2]\
        ,based_periodicity[5]))

    common_periodicity=tuple(map(lambda x: int(round(x)),common_periodicity))

    # print("common_periodicity ",common_periodicity)
    return common_periodicity


def construct_configuration_based_on_target_periodicty(target_periodicity,based_configuration_class,species_prim):
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


    constructed_spin_configuration_class=spin_configuration_class(constructed_spin_configuration,target_periodicity,species_prim)
    return  constructed_spin_configuration_class

    ## need to do some mathemtical checking, done




def construct_configuration_based_on_target_periodicty_batch(target_periodicity,based_configuration_class,species_prim,trialsize=100):
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


    # constructed_spin_configuration={}

    constructed_spin_configuration_batch=[]
    constructed_spin_configuration_class_batch=[]
    for i in range(trialsize):
        constructed_spin_configuration_batch.append({})

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

                    sum_now=xs_to_check*ys_to_check*zs_to_check
                    # element_selectecd=''
                    element_selectecd_list=[]

                    for specie_now in possible_species:
                        possible_species[specie_now]/=float(sum_now)


                    for trial_now in range(trialsize):

                        random_number=random.random()
                        to_break=False
                        # print(random_number)
                        # print(possible_species)
                        culmulated = 0

                        for specie_now in possible_species:
                            # possible_species[specie_now]/=float(sum_now)

                            if random_number >= culmulated and random_number<= culmulated + possible_species[specie_now]:
                                element_selectecd_list.append( specie_now)
                                to_break=True
                            culmulated=culmulated + possible_species[specie_now]
                            if to_break:
                                break
                        # print(element_selectecd_list)
                        assert len(element_selectecd_list)==trial_now+1

                        constructed_spin_configuration_batch[trial_now][original_index]=element_selectecd_list[trial_now]
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

    for i in range(trialsize):
        constructed_spin_configuration_class_batch.append(spin_configuration_class(constructed_spin_configuration_batch[i],target_periodicity,species_prim))
    return  constructed_spin_configuration_class_batch

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

    inverse_configuration_now_removing_vacs=spin_configuration_class.inverse_configuration.copy()

    if "Vac" in inverse_configuration_now_removing_vacs:
        del inverse_configuration_now_removing_vacs["Vac"]


    file_content+="generated from extract small configs from GMC\n"
    file_content+="1.0\n"
    file_content+="{:.8f}{:17.8f}{:17.8f}\n".format(target_primitive_x[0],target_primitive_x[1],target_primitive_x[2])
    file_content+="{:.8f}{:17.8f}{:17.8f}\n".format(target_primitive_y[0],target_primitive_y[1],target_primitive_y[2])
    file_content+="{:.8f}{:17.8f}{:17.8f}\n".format(target_primitive_z[0],target_primitive_z[1],target_primitive_z[2])
    # file_content+=spin_configuration_class.element_count
    file_content+='  '.join(inverse_configuration_now_removing_vacs.keys())+"\n"
    file_content+='  '.join([str(len(x)) for x in \
        inverse_configuration_now_removing_vacs.values()])+"\n"
    file_content+='direct\n'

    # print(" spin_configuration_class.inverse_configuration is")
    # print( spin_configuration_class.inverse_configuration)
    for (keys,values) in  inverse_configuration_now_removing_vacs.iteritems():
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


    if not os.path.exists(os.path.dirname(destination)):
        try:
            os.makedirs(os.path.dirname(destination))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise



    f=open(destination,'w')
    f.write(file_content)
    f.close()

    pass


def remove_batch_duplicates(constructed_spin_configuration_class_batch):
    to_delete_list=[]
    for i in range(1,len(constructed_spin_configuration_class_batch)):
        i_is_duplicate=False
        for j in range(0,i):
            if j in to_delete_list:
                continue
            compare_result=compare_spin_config_class(constructed_spin_configuration_class_batch[j],constructed_spin_configuration_class_batch[i])
            if compare_result == True:
                i_is_duplicate = True
                break

        if i_is_duplicate:
            to_delete_list.append(i)

    constructed_spin_configuration_class_batch_new=[v for i,v in enumerate(constructed_spin_configuration_class_batch) if i not in to_delete_list]
    return constructed_spin_configuration_class_batch_new



def compare_spin_config_class(config_a,config_b):

    are_they_the_same=None
    conc_diff=compare_conc(config_a,config_b)
    if conc_diff<1e-7:
        conc_the_same=True
    else:
        conc_the_same=False

    if conc_the_same==False:
        are_they_the_same=False
        return are_they_the_same

    if config_a.periodicity != config_b.periodicity or config_a.subcell_number != config_b.subcell_number:
        are_they_the_same=False
        return are_they_the_same

    subcell_number=config_a.subcell_number
    periodicity=config_a.periodicity

    they_are_the_same_for_some_translation=False
    break_first_3=False
    for i in range(periodicity[0]):
        for j in range(periodicity[2]):
            for k in range(periodicity[5]):

                they_are_the_same_for_this_translation=True
                break_last_4=False
                for i0 in range(periodicity[0]):

                    for j0 in range(periodicity[2]):

                        for k0 in range(periodicity[5]):

                            for p0 in range(subcell_number):
                                i1=(i+i0)%periodicity[0]
                                j1=(j+j0)%periodicity[2]
                                k1=(k+k0)%periodicity[5]
                                p1=p0
                                if config_a.configuration[i0,j0,k0,p0]!=config_b.configuration[i1,j1,k1,p1]:
                                    they_are_the_same_for_this_translation=False
                                    break_last_4=True

                                if break_last_4:
                                    break
                            if break_last_4:
                                break
                        if break_last_4:
                            break
                    if break_last_4:
                        break
                if they_are_the_same_for_this_translation==True:
                    they_are_the_same_for_some_translation=True
                    break_first_3=True

                if break_first_3:
                    break
            if break_first_3:
                break
        if break_first_3:
            break


    return they_are_the_same_for_some_translation




def compare_conc(config_a,config_b):
#return the absolute diff of concentration
    conc_a=config_a.calculate_concentration()
    conc_b=config_b.calculate_concentration()

    abs_diff=0

    for ele,conc_a_ele_now in conc_a.iteritems():
        if ele not in conc_b:
            abs_diff+=conc_a_ele_now
        else:
            abs_diff+= abs(conc_a_ele_now-conc_b[ele])

    for ele,conc_b_ele_now in conc_a.iteritems():
        if ele not in conc_a:
            abs_diff+=conc_b_ele_now

    return abs_diff



def remove_batch_duplicates_pymatgen(constructed_spin_configuration_class_batch,avec_prim,coords_prim,species_prim,output_poscar_destination,up_to_how_many_unique_struct):
    output_poscar_destination_pymatgen_tmp=output_poscar_destination+"_pymatgen_tmp_POSCAR"

    print("in remove_batch_duplicates_pymatgen")

    constructed_spin_configuration_class_batch_pymatgen=[]
    for i in range(len(constructed_spin_configuration_class_batch)):
        if os.path.exists(output_poscar_destination_pymatgen_tmp):
            os.remove(output_poscar_destination_pymatgen_tmp)
        print_to_poscar(constructed_spin_configuration_class_batch[i],avec_prim,coords_prim,species_prim,output_poscar_destination_pymatgen_tmp)
        constructed_spin_configuration_class_batch_pymatgen.append(pymatgen.Structure.from_file(output_poscar_destination_pymatgen_tmp))
        # print(constructed_spin_configuration_class_batch_pymatgen[i])

    if os.path.exists(output_poscar_destination_pymatgen_tmp):
        os.remove(output_poscar_destination_pymatgen_tmp)


    to_delete_list=[]
    mg=StructureMatcher()
    for i in range(1,len(constructed_spin_configuration_class_batch_pymatgen)):
        if up_to_how_many_unique_struct <= i - len(to_delete_list):
            break

        i_is_duplicate=False
        for j in range(0,i):
            if j in to_delete_list:
                continue
            compare_result=mg.fit(constructed_spin_configuration_class_batch_pymatgen[j],constructed_spin_configuration_class_batch_pymatgen[i])

            if compare_result == True:
                i_is_duplicate = True
                print(i," is a duplcate of ",j)
                break
            print(i," is not a duplcate of ",j)

        if i_is_duplicate:
            to_delete_list.append(i)

    constructed_spin_configuration_class_batch_new=[v for i,v in enumerate(constructed_spin_configuration_class_batch) if i not in to_delete_list]
    return constructed_spin_configuration_class_batch_new





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
        help="Path to POSCAR file.",
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


    parser.add_argument(
        "--BatchSize","-batch",
        help="how many POSCAR output to produce .",
        default=None,
        type=int
    )

    parser.add_argument(
        "--TrialMultiplier","-trialmltp",
        help="How many trials based on the multiple of BatchSize.",
        default=5,
        type=int
    )

    parser.add_argument(
        "--SaveErrorVectorObjFile",
        help="Path to write Error Vector .",
        type=str
    )

    parser.add_argument(
        "--LoadErrorVectorObjFile",
        help="Path to load Error Vector .",
        type=str
    )



    args = parser.parse_args()
    maxsize=float(args.MAXSIZE)+0.5
    output_poscar_destination=args.WriteToPoscar
    SaveErrorVectorObjFile = args.SaveErrorVectorObjFile
    LoadErrorVectorObjFile = args.LoadErrorVectorObjFile

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

        supercell_index=np.floor( freactional_coords_according_to_PRIM+1e-2)
        # print("supercell_index")
        # print(supercell_index)

        subcell_position= freactional_coords_according_to_PRIM-supercell_index
        # print("subcell_position")
        # print(subcell_position)

        residual_position=np.abs(coords_prim-subcell_position)
        # print("residual_position")
        # print(residual_position)

        sum_error=np.sum(residual_position,1)
        # print ("sum_error")
        # print (sum_error)

        subcell_index=np.argmin(sum_error)
        assert np.amin(sum_error)<1e-2

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


    spin_configuration_poscar_class=spin_configuration_class(configurations_poscar,periodicity_poscar,species_prim)


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

    if LoadErrorVectorObjFile is None:
        for i in range(len(periodicity_vector_list)):
            if i%100 == 0:
                print ("calculated ",i," periodicity error vector")
            # if i == 10:
            #     break
            periodicity_vector_now=periodicity_vector_list[i]
            # print ("periodicity_vector_now",periodicity_vector_now)
            # print ("spin_configuration_poscar_class")
            # print (spin_configuration_poscar_class)
            error_rate_now=calculate_error_rate_for_this_periodicity_vector(periodicity_vector_now,spin_configuration_poscar_class)
            error_rate_list.append(error_rate_now)
    else:
        error_rate_list = load_obj(LoadErrorVectorObjFile)

    if LoadErrorVectorObjFile is not None and SaveErrorVectorObjFile is not None:
        raise ("Please do not use LoadErrorVectorObjFile and SaveErrorVectorObjFile at the same time")

    if LoadErrorVectorObjFile is None and SaveErrorVectorObjFile is not None:
        save_obj(error_rate_list,SaveErrorVectorObjFile)

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



    batch_production = True
    if not batch_production:
        constructed_spin_configuration_class=construct_configuration_based_on_target_periodicty(target_periodicity,spin_configuration_poscar_class, species_prim)

        # print(spin_configuration_poscar_class.concentration)
        print("input concentration")
        print(spin_configuration_poscar_class.concentration)
        print("output concentration")
        print(constructed_spin_configuration_class.concentration)

        print_to_poscar(constructed_spin_configuration_class,avec_prim,coords_prim,species_prim,output_poscar_destination)


    # constructed_spin_configuration_class_max=construct_configuration_based_on_target_periodicty(target_periodicity,spin_configuration_poscar_class,species_prim)

    if batch_production:
        constructed_spin_configuration_class_batch=construct_configuration_based_on_target_periodicty_batch(target_periodicity,spin_configuration_poscar_class,species_prim,trialsize=args.BatchSize*args.TrialMultiplier)




    # pprint(constructed_spin_configuration_class_batch[0].configuration)
    # print(constructed_spin_configuration_class_batch[0].periodicity)
    # print(constructed_spin_configuration_class_batch[0].subcell_number)


    print("before removing duplicacy")
    print("input concentration")
    print(spin_configuration_poscar_class.concentration)
    print("output batch concentration")
    for i in range(len(constructed_spin_configuration_class_batch)):
        print(constructed_spin_configuration_class_batch[i].concentration)


    constructed_spin_configuration_class_batch=remove_batch_duplicates(constructed_spin_configuration_class_batch)

    conc_dff_list=[]
    for i in range(len(constructed_spin_configuration_class_batch)):
        conc_dff_list.append(compare_conc(constructed_spin_configuration_class_batch[i],spin_configuration_poscar_class))

    zip_list= zip(constructed_spin_configuration_class_batch,conc_dff_list)
    zip_list.sort(key = lambda t: t[1])

    constructed_spin_configuration_class_batch=[zip_list[i][0] for i in range(len( constructed_spin_configuration_class_batch))]


    print("after removing duplicacy based on translation")
    print("input concentration")
    print(spin_configuration_poscar_class.concentration)
    print("output batch concentration")
    for i in range(len(constructed_spin_configuration_class_batch)):
        print(constructed_spin_configuration_class_batch[i].concentration)



    constructed_spin_configuration_class_batch=remove_batch_duplicates_pymatgen(constructed_spin_configuration_class_batch,avec_prim,coords_prim,species_prim,output_poscar_destination,
                                                                                up_to_how_many_unique_struct=args.BatchSize)

    print("after sorting and removing duplicacy based on pymatgen up to batch size of ",args.BatchSize)
    print("input concentration")
    print(spin_configuration_poscar_class.concentration)
    print("output batch concentration")
    for i in range(len(constructed_spin_configuration_class_batch)):
        print(constructed_spin_configuration_class_batch[i].concentration)



    constructed_spin_configuration_class_batch=[constructed_spin_configuration_class_batch[i] for i in range(min(args.BatchSize,len(constructed_spin_configuration_class_batch)))]

    print("after selecting best matched conc")
    print("input concentration")
    print(spin_configuration_poscar_class.concentration)
    print("output batch concentration")
    for i in range(len(constructed_spin_configuration_class_batch)):
       print(constructed_spin_configuration_class_batch[i].concentration)


    for i in range(len(constructed_spin_configuration_class_batch)):
        i_str=str(i).zfill(2)
        print_to_poscar(constructed_spin_configuration_class_batch[i],avec_prim,coords_prim,species_prim,output_poscar_destination+"_"+i_str)

        # print("config ",i,"is")
        # pprint(constructed_spin_configuration_class_batch[i].configuration)
        # pprint(constructed_spin_configuration_class_batch[i].inverse_configuration)
        # print(constructed_spin_configuration_class_batch[i].periodicity)
        # print(constructed_spin_configuration_class_batch[i].subcell_number)


