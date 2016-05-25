#Written by Yulia Emelyanova
#Code for getting the style ID information from the prices
#!/usr/bin/python                                                                                           
import urllib, urllib2
import httplib
from urlparse import urlparse
import time
urlmodel="accord"

#open the text file with the prices 
r=open('prices.txt','r')
u=open('style_info.txt', 'wb')
print "program in progress"
count=0


# for every line in the file take the style ID and put it in url for style retrieval
for line in r:
    check="ID"
    if check in line:
        words=line.rsplit(":", 1)[1]
        words=words.strip()
        url2="https://api.edmunds.com/api/vehicle/v2/styles/"+ words +"?fmt=json&api_key=eu5jm79n56bk3y33u5hdpt65"
        user_agent="Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
        headers={"User-Agent": user_agent}

        try:
            request=urllib2.Request(url2, None, headers)
            response=urllib2.urlopen(request)
            print "GOT STYLE ID"
            content=response.read()
            u.write(content)
            count+=1
            if count==10:
                time.sleep(3)
                count=0
            else:
                continue
                             
        except urllib2.HTTPError, e:
            if e.code==404:
                print "SOMETHING WRONG"
                
    else:
        pass
