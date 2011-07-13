#!/usr/bin/env python
import os, os.path, shutil, sys, re, fileinput

vptfile= open("boost/numeric/mtl/interface/vpt.hpp")
groupfile= open("vampir_groups.dat", "w")

l=['template <> std::string vampir_trace<2001>::name("gen_vector_copy");',
   'template <> std::string vampir_trace<2002>::name("cross");', 'Quatsch']

group_names=['mtl_utilities', 'mtl_fsize', 'mtl_vector', 'mtl_vecmat', 'mtl_matrix', 'mtl_factor', 'mtl_solver', '', '', 'mtl_app']
groups=[[],[],[],[],[],[],[],[],[],[]]
assert len(group_names) == len(groups)

pat= re.compile('template <> std::string vampir_trace<(\d).+>::name\("(.*)"\);')

for s in vptfile.readlines() : 
    mm= pat.search(s)
    if mm and mm.group(2):
        groups[int(mm.group(1))].append(mm.group(2))

# print groups

for i in range(len(groups)):
    if group_names[i]:
        groupfile.write(group_names[i]+"="+";".join(groups[i])+'\n')
        #print group_names[i]+"="+";".join(groups[i])
groupfile.close()
