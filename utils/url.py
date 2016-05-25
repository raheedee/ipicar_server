#Written by Yulia Emelyanova
#Takes the main car file and retrieves all the style ID information about every individual car and saves it into a seperate text file
#!/usr/bin/python
import urllib, urllib2
import time
urlmodel="accord"

makes=open('carpiece.txt', 'r')
f=open('styles_ex.txt','w')
print "program in progress"
count=0
count_total=0
#identifies the makes, model and year and puts it into the url for style ID retrieval
for line in makes:
    if "id" in line:
        words=line.split()
        words=str(words[1])
        words=words.rsplit(",",1)[0]
        if words.isdigit():
            line=next(makes)
            check="year"
            if check in line:
                pass
            else:
                line2=next(makes)
                make=line2.rsplit(":", 1)[1]
                make=make.partition('"')[-1].rpartition('"')[0]
            
        else:
            line=next(makes)
            check="year"
            if check in line:
                pass
            else:
                line3=next(makes)
                model_name=line3.rsplit(":",1)[1]
                model_name=model_name.partition('"')[-1].rpartition('"')[0]
                
            
        
    if "year" in line:
        nums=line.split()
        if nums[1]!="[":
            urlyear=str(nums[1])
            quoted_make=urllib.quote(make)
            quoted_model=urllib.quote(model_name)
            url="https://api.edmunds.com/api/vehicle/v2/"+ quoted_make + "/" + quoted_model+ "/" + urlyear +"/styles?fmt=json&api_key=wbsyqcnuetaucpk53mbn4yfv"
            print url
            user_agent="Mozilla/5.0 (Windows NT 6.1; Win64; x64)"
            headers={"User-Agent": user_agent}
            request=urllib2.Request(url, None, headers)
            response=urllib2.urlopen(request)
            content=response.read()
            f.write(content)
            count +=1
            count_total+=1
            if count==10:
                time.sleep(3)
                count=0
            else:
                continue
        
            
                
f.close()
makes.close()
