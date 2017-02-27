# -*- coding:utf-8 -*-
# This script will download all the 10-K, 10-Q and 8-K
# provided that of company symbol and its cik code.

import requests
import os
import errno
from bs4 import BeautifulSoup
from config import FORMS_DATA_PATH, HTTP_TIMEOUT
import urllib2


class SecCrawler():

    def __init__(self):
        self.hello = "Welcome to Sec Cralwer!"
        self.failed_urls = []
        self.fail_flag = False
        self.fail_count = 0
        self.max_attempts = 20
        print("Path of the directory where data will be saved: " + FORMS_DATA_PATH)

    def make_directory(self, company_code, cik, priorto, filing_type):
        # Making the directory to save comapny filings
        path = os.path.join(FORMS_DATA_PATH, company_code, cik, filing_type)

        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

    def save_in_directory(self, company_code, cik, priorto, doc_list,
        doc_name_list, filing_type):
        # Download and save every text document into its respective folder
        for j in range(len(doc_list)):
            base_url = doc_list[j]
            for a in range(self.max_attempts):
                try:
                    r = requests.get(base_url, timeout=HTTP_TIMEOUT)
                    data = r.text
                    path = os.path.join(FORMS_DATA_PATH, company_code, cik,
                    filing_type, doc_name_list[j])

                    with open(path, "a+") as f:
                        f.write(data.encode('ascii', 'ignore'))
                #except requests.exceptions.ReadTimeout:
                except urllib2.HTTPError as err:
                    if err.code == 429:
                        print "429 HTTP Error, sleeping for an hour..."
                        time.sleep(1200)
                        print "Back online!"
                    elif err.code == 404:
                        print cik + "\nError 404, failed to pull doc list from: " + base_url
                        self.fail_flag = True
                        self.fail_count = self.fail_count + 1
                        self.failed_urls.append(base_url)
                        failpath = os.path.join(FORMS_DATA_PATH, 'failed_urls.txt')
                        with open(failpath, "a+") as fp:
                            fp.write(cik + ',' + base_url + '\n')
                    else:
                        typ, value, traceback = sys.exc_info()
                        print "Could not retrieve doc list for " + cik + '\n' 
                        print str(typ)
                        print str(value)
                        print "\n Retrying. Attempt " + str(a)
                except requests.exceptions.Timeout as e:
                    print e
                    print "HTTP Read timeout on doc list, retry number " + str(a)
                except urllib3.exceptions.ConnectTimeoutError:
                	print "HTTP Connections to EDGAR exceeded. Sleeping for 11 minutes..."
                	time.sleep(660)
                	print "Back online"
                else:
                    break
                finally:
                    if a == self.max_attempts - 1:
                        print cik + "\nExceeded attempts, failed to pull doc list from: " + base_url
                        self.fail_flag = True
                        self.fail_count = self.fail_count + 1
                        self.failed_urls.append(base_url)
                        failpath = os.path.join(FORMS_DATA_PATH, 'failed_urls.txt')
                        with open(failpath, "a+") as fp:
                            fp.write(cik + ',' + base_url + '\n')   

    def filing_4(self, company_code, cik, afterdate, priorto):

        self.make_directory(company_code,cik, priorto, 'form4')

        # generate the url to crawl
        start_doc = 1
        more_docs = True

        while more_docs:
            base_url = "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="+str(cik)+"&type=4" + "&datea=" + str(afterdate) + "&dateb="+str(priorto)+"&owner=include&output=xml&start=" + str(start_doc) + "&count=100"
            print ("Fetching doc list for " + str(company_code))
            for attempts in range(self.max_attempts):
                try:
                    r = requests.get(base_url, timeout=HTTP_TIMEOUT)
                #except requests.exceptions.ReadTimeout:
                except urllib2.HTTPError as err:
                    if err.code == 429:
                        print 
                        print "429 HTTP Error, sleeping for an 20 minutes..."
                        time.sleep(1200)
                        print "Back online!"
                    elif err.code == 404:
                        print "Error 404, doc unavailable, skipping doc for" + cik
                        failpath = os.path.join(FORMS_DATA_PATH, 'failed_urls.txt')
                        with open(failpath, "a+") as fp:
                            fp.write("404 Error. Failed " + cik + "at: " + base_url + '\n')
                        return
                    else:
                        typ, value, traceback = sys.exc_info()
                        print "Could not retrieve SEC data\n" 
                        print str(typ)
                        print str(value)
                        print "\n Retrying. Attempt " + str(attempts)
                except requests.exceptions.Timeout as e:
                    print e
                    print "HTTP Read timeout, retry number " + str(attempts)
                except urllib3.exceptions.ConnectTimeoutError:
                	print "HTTP Connections to EDGAR exceeded. Sleeping for 11 minutes..."
                	time.sleep(660)
                	print "Back online"
                else:
                    break
                finally: 
                    if attempts == self.max_attempts - 1:
                        print 'Failed to retrieve doc for' + cik
                        with open(failpath, "a+") as fp:
                            fp.write("Too many attempts. Failed " + cik + "at: " + base_url + '\n')
                        return
            data = r.text

            # get doc list data
            doc_list, doc_name_list = self.create_document_list(data)

            try:
                self.save_in_directory(company_code, cik, priorto, doc_list, doc_name_list, 'form4')
            except Exception as e:
                print (str(e))

            more_docs = True if doc_list else False
            start_doc = start_doc + 100
        if not self.fail_flag:
            print ("Successfully downloaded all the files for " + str(cik))
        else:
            print ("Failed to download " + str(self.fail_count) + " files: " + str(self.failed_urls))
            self.fail_flag = False
            self.fail_count = 0

    def create_document_list(self, data):
        # parse fetched data using beatifulsoup
        print "about to create soup object"
        soup = BeautifulSoup(data)
        # store the link in the list
        link_list = list()

        # If the link is .htm convert it to .html
        for link in soup.find_all('filing'):
            url = link.filinghref.string
            if link.filinghref.string.split(".")[len(link.filinghref.string.split("."))-1] == "htm":
                url += "l"
            if link.type.string == '4':
                link_list.append(url)
        link_list_final = link_list

        print ("Number of files to download {0}".format(len(link_list_final)))
        print ("Starting download....")

        # List of url to the text documents
        doc_list = list()
        # List of document names
        doc_name_list = list()

        # Get all the doc
        for k in range(len(link_list_final)):
            required_url = link_list_final[k].replace('-index.html', '')
            txtdoc = required_url + ".txt"
            docname = txtdoc.split("/")[-1]
            doc_list.append(txtdoc)
            doc_name_list.append(docname)
        return doc_list, doc_name_list

