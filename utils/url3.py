#Written by Yulia Emelyanova
#The code that takes the acquired style ID's from the text file and using the Style ID pulls either the letter reviews or user reviews for that 
#particular car, it also keeps track of cars that have neither of those reviews
#!/usr/bin/python                                                                                           
import urllib, urllib2
import httplib
from urlparse import urlparse
import time
urlmodel="accord"

#open the Style ID file and create three text files that will store the letter reviews, user reviews and ID's that contained no reviews
n=open('noreviews.txt', 'w')
s=open('example.txt', 'r')
r=open('reviewsbystyle.txt','wb')
u=open('userreviewsbystyle.txt', 'wb')
print "program in progress"
#keep the count of how many reviews were retrieved
count=0
count_total=0
user_reviews=0
letter_reviews=0
no_reviews=0

#for every line get the Style ID within the file and paste it into the letter grade retrieval url            
for line1 in s:
    if "id" in line1:
        words=line1.split() 
        words=str(words[1])                                                                        
        words=words.rsplit(",",1)[0]
        
        if words.isdigit():

            s.next()
            line2=next(s)
            
            check="year"                                                    
            if check in line2:
                sub=words
               
                quoted_subs=urllib.quote(sub)
                url2= "https://api.edmunds.com/api/vehicle/v2/styles/" + quoted_subs +"/grade?fmt=json&api_key=weg9cm94a4qa3u27nymzevq4"
                print url2
                user_agent="Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
                headers={"User-Agent": user_agent}
                
                try:
                    request=urllib2.Request(url2, None, headers)
                    response=urllib2.urlopen(request)
                    print "GOT LETTER REVIEW"
                    content=response.read()
                    r.write(content)
                    count+=1
                    
                    letter_reviews+=1
                    if count==10:
                        time.sleep(3)
                        count=0
                    else:
                        continue
                #if it is successful store the review and add the counter if not try to see if this car has a user review
                except urllib2.HTTPError, e:
                    if e.code == 404:
                        print "URL DOES NOT EXIST"
                        url3="https://api.edmunds.com/api/vehiclereviews/v2/styles/"+ quoted_subs+"?fmt=json&api_key=weg9cm94a4qa3u27nymzevq4"
                        user_agent="Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
                        headers={"User-Agent": user_agent}
                        try:
                            requesting=urllib2.Request(url3, None, headers)
                            response1=urllib2.urlopen(requesting)
                            print "GOT USER REVIEW"
                            content=response1.read()
                            u.write(content)
                            count+=1
                            
                            user_reviews+=1
                            if count==10:
                                time.sleep(3)
                                count=0
                            else:
                                continue
                        #if there is no user review then add a counter for no reviews to keep track
                        except urllib2.HTTPError, e:
                            if e.code==404:
                                print "NO USER REVIEW"
                                no_reviews+=1
                                n.write(url3 + '\n')
                            
            
            else:
                pass

print "number of user reviews:%s"% user_reviews
print "number of letter reviews:%s"% letter_reviews
print "number of no reviews:%s"% no_reviews

s.close()
r.close()
u.close()
n.close()                

