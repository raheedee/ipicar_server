#Yulia Emelyanova
#Code for getting additional prices
#!/usr/bin/python                                                                                                            
import urllib, urllib2
import time
urlmodel="accord"

#open the file containing the ID's of cars that did not initially contain prices
makes=open('noprices.txt', 'r')
f=open('extraprice.txt','w')
print "program in progress"
count=0
count_total=0
#take the Style ID from the file and pull the information
for line in makes:
    print line
    if "Style ID" in line:
        year=line.rsplit(":", 1)[1]
        year=year.strip()
        url="https://api.edmunds.com/v1/api/tco/newtotalcashpricebystyleidandzip/"+ year+ "/35223?fmt=json&api_key=wjtr9qcqcevm8z49ydcebdyz"
        print url
        user_agent="Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
        headers={"User-Agent": user_agent}

        try:
            request=urllib2.Request(url, None, headers)
            response=urllib2.urlopen(request)
            print "GOT PRICE!!!"
            content=response.read()
            f.write(content)
            f.write('Style ID: '+ year + '\n')
            count+=1
    
            if count==10:
                time.sleep(3)
                count=0
            else:
                continue
    
        except urllib2.HTTPError as e:
            if e.code==400:
                print "NO PRICE LISTED!!!"
                   
                count+=1
                                                                                        
                if count==10:
                    time.sleep(3)
                    count=0
                else:
                    continue

