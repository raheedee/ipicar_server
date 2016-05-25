#Written by Yulia Emelyanova
#goes through the car ID's without reviews and shows what the year range is, this was originally for our filter purposes when we were deciding 
#whether we should include prices in the filter or not
#!/usr/bin/python
import urllib, urllib2
import time
urlmodel="accord"

makes=open('Years.txt', 'r')
print "program in progress"
lesscount=0
twozerocount=0
twoonecount=0
twotwocount=0
twothreecount=0
twofourcount=0
twofivecount=0
greatercount=0
unknown=0
#open the file and for every year identify where it belongs, keeping a counter of how many cars fall under a particular year
for line in makes:
    if "id" in line:
        line2=next(makes)
        check="year"
        if check in line2:
            year=line2.rsplit(":", 1)[1]
            year=year.strip()
            if year.isdigit():
                year=int(year)
                print year
                if year<2000:
                    lesscount +=1
                elif year==2000:
                    twozerocount+=1
                elif year==2001:
                    twoonecount+=1
                elif year==2002:
                    twotwocount+=1
                elif year==2003:
                    twothreecount+=1
                elif year==2004:
                    twofourcount+=1
                elif year==2005:
                    twofivecount+=1
                elif year>2005:
                    greatercount+=1
                
        else:
            continue


#print all the final count information
print "number of cars less than 2000:%s"% lesscount
print "number of cars in 2000:%s"% twozerocount
print "number of cars in 2001:%s"% twoonecount
print "number of cars in 2002:%s"% twotwocount
print "number of cars in 2003:%s"% twothreecount
print "number of cars in 2004:%s"% twofourcount
print "number of cars in 2005:%s"% twofivecount
print "number of cars released after 2005:%s"% greatercount        

makes.close()
