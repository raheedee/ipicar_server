#Yulia Emelyanova
#Code for building the url meant to retrieve pictures manually
#!/usr/bin/python                                                                                       
                                                                                                                    
import urllib, urllib2
import time
urlmodel="accord"

#open the text file with all the style information
makes=open('all_the_styles.txt', 'r')
f=open('links.txt','w')

# for each line pull the information needed to build the url; make, model, year, submodel, and trim
for line in makes:
    if "make" in line:
        line=next(makes)
        line=next(makes)
        line=next(makes)
        line2=line.rsplit(":",1)[1]
        make=line2.partition('"')[-1].rpartition('"')[0]
        
    if "model" in line:
        line=next(makes)
        line=next(makes)
        line=next(makes)
        line2=line.rsplit(":",1)[1]
        model=line2.partition('"')[-1].rpartition('"')[0]
        
    if "year" in line:
        take=line.rsplit(":", 1)[1]
        if take.isdigit():
            print take
        else:
            pass
    if "submodel" in line:
        line=next(makes)
        line=next(makes)
        line2=line.rsplit(":",1)[1]
        sub=line2.partition('"')[-1].rpartition('"')[0]
    
    if "trim" in line:
        trim=line.rsplit(":", 1)[1]
        trim=trim.partition('"')[-1].rpartition('"')[0]
        #put the url all together
        url="https://media.ed.edmunds-media.com/"+ make +"/"+ model +"/"+take+"/oem/"+take+"_"+make+"_"+model+"_"+sub+"_"+trim+"_fq_oem_1_500.jpg"
        print url
