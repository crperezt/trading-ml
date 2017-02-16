# -*- coding:utf-8 -*-
# This script will generate insider datasets from Form 4 data

import requests
from os import listdir
from os.path import isdir, join
import errno
from bs4 import BeautifulSoup
from config import FORMS_DATA_PATH
import pandas as pd
import sys


class SecCrawler():

    def __init__(self):
        self.hello = "Welcome to Sec Crawler!"
    
    #def parse_form4(self, company_code, start_date, end_date):
    def parse_form4(self, start_date, end_date, returns=False):
        #start_date, end_date in YYYY-MM format
        #Iterate through files for company
        #Calculate monthly NBC, NBV from start_date to end_date
        #Generate list of 12-tuples with month-year
        #Fetch return for months following month-year of each 12-tuple
        #Generate datapoints for company:
         #COMPANY, 12-month NBC 12-tuple, 12-month NBV 12-tuple, Sector, MktCap, Return for month 13
        
        #get list of all companies in SEC data directory
        #company_list = [f for f in listdir(DEFAULT_DATA_PATH) if isdir(join(DEFAULT_DATA_PATH,f))]
        company_list = pd.read_csv(join(FORMS_DATA_PATH, 'company_list.csv'))

        #generate list of months to include in data
        start_month = int(start_date[5:7])
        start_year = int(start_date[0:4])
        end_month = int(end_date[5:7])
        end_year = int(end_date[0:4])
        date_set = [] 
        while start_year <= end_year:
            if start_year == end_year:
                if start_month > end_month:
                    break
            if start_month < 10:
                date_set.append(str(start_year) + '-0' + str(start_month))
            else:
                date_set.append(str(start_year) + '-' + str(start_month))
            start_year = start_year + 1 if start_month == 12 else start_year
            start_month = start_month + 1 if start_month != 12 else 1
        date_set.sort()

        dataset_file = open(join(FORMS_DATA_PATH, 'picks_dataset.csv'), 'a+')
        dataset_file.write('COMPANY,NBC1,NBC2,NBC3,NBC4,NBC5,NBC6,NBC7,NBC8,NBC9,NBC10,NBC11,NBC12,NBV1,NBV2,NBV3,NBV4,NBV5,NBV6,NBV7,NBV8,NBV9,NBV10,NBV11,NBV12,SECTOR,MKTCAP,RETURN\n')
        
        #for each company c directory
        #traverse through all downloaded files
        for i, c in enumerate(company_list.SYMBOL):
            print "Processing " + c + ". Company " + str(i) + " of " + str(len(company_list.SYMBOL))
            cpath = join(FORMS_DATA_PATH, c, c, 'form4')
            csector = company_list.SECTOR[i]
            cmktcap = company_list.MKTCAP[i]
            file_list = [f for f in listdir(cpath)]
            if not file_list:
                continue
            
            #create dictionary for data
            #keyed by month, each entry will have the totals of transactions
            #for that month
            company_data = {}
            for t in date_set:
                company_data[t] = {'purch': 0, 'sales': 0, 'sb': 0, 'ss': 0}
            
            #for each file within a company directory
            #find all transactions and aggregate them by month
            #input the totals into company_data dictionary
            for j, f in enumerate(file_list):
                form = open(join(cpath,f))
                #print join(cpath,f)

                soup = BeautifulSoup(form.read(), 'lxml-xml')
                #find all transactions in file
                for t in soup.find_all('nonDerivativeTransaction'):
                    t_date = t.transactionDate.value.string[0:7]
                    if t_date not in date_set:
                        #print "Date out of range!"
                        continue
                    num_shares = int(float(t.transactionShares.value.string))
                    if t.transactionAcquiredDisposedCode.value.string == 'A':
                        company_data[t_date]['sb'] = company_data[t_date]['sb'] + num_shares
                        company_data[t_date]['purch'] = company_data[t_date]['purch'] + 1
                    elif t.transactionAcquiredDisposedCode.value.string == 'D':
                        company_data[t_date]['ss'] = company_data[t_date]['sb'] + num_shares
                        company_data[t_date]['sales'] = company_data[t_date]['sales'] + 1
       
            #sort the dictionary by date for dataset
            #to add return data for 12-month segments
            sorted_cd = company_data.items()
            sorted_cd.sort()
            #print "Sorted company data for " + c + "\n" + str(sorted_cd) + "\n"

            #create dataset for company c
            #for each month, form tuple with data of next 11 months
            #and form a datapoint
            #dataset_file.write('COMPANY,NBC,NBV,SECTOR,MKTCAP,RETURN\n')
            for j, cd in enumerate(sorted_cd):
                #if there are no 12 consecutive months after this month
                #finish
                if j + 12 > len(sorted_cd):
                    break
                #if this month has no trades, skip
                #if sorted_cd[j][1]['purch'] == 0 and sorted_cd[j][1]['sales'] == 0:
                #    continue

                
                dataset_file.write(c + ',')
                nbc = []
                nbv = []
                for k in range (j,j+12):
                    #print "Computing stats for " + c + " for month " + str(sorted_cd[k][0])
                    try:
                        nbc_k = float(sorted_cd[k][1]['purch'] - sorted_cd[k][1]['sales'])/float(sorted_cd[k][1]['purch'] + sorted_cd[k][1]['sales'])                      
                    except ZeroDivisionError:
                        nbc.append(0.0)
                    else:
                        nbc.append(nbc_k)
                    try:
                        nbv_k = float(sorted_cd[k][1]['sb'] - sorted_cd[k][1]['ss'])/float(sorted_cd[k][1]['sb'] + sorted_cd[k][1]['ss']) 
                    except ZeroDivisionError:
                        nbv.append(0.0)
                    else:
                        nbv.append(nbv_k)
                
                for n in nbc:
                    dataset_file.write(str(n) + ',')
                for n in nbv:
                    dataset_file.write(str(n) + ',')

                #fetch return for month 13
                if returns:
                    ret = self.get_return_13(c, sorted_cd[j][0])
                    dataset_file.write(str(csector) + ',' + str(cmktcap) + ',' + str(ret) + '\n')
                else:
                    dataset_file.write(str(csector) + ',' + str(cmktcap) + '\n')

        dataset_file.close()
            


    def get_return_13(self, company, month):

        month_int = int(month[5:7])
        month_first = month + '-01'

        last_days = {1: '31', 2: '28', 3: '31', 4: '30', 5: '31', 6: '30',
        7:'31', 8:'31', 9:'30', 10:'31', 11:'30', 12:'31'}

        month_last = month + '-' + last_days[month_int]

        #stock = Share(company)
        for attempts in range (20):
            try:
                ret_list = pd.read_json("https://api.tiingo.com/tiingo/daily/" + company.lower() + "/prices?startDate=" + month_first + "&endDate=" + month_last + "&token=8a387055f2f4081b89abfc6b3044284e958f178e")
            except:
                typ, value, traceback = sys.exc_info()
                print "Could not retrieve stock data\n" 
                print str(typ)
                print str(value)
                print "\n Retrying..."    
            else:
                break
            finally:
                if attempts == 19:
                    print "Failed to fetch stock data\n"
                    return 'Fail'


        #print ret_list.head()
        #print ret_list.tail()
        try:
            ret = (float(ret_list.adjClose[len(ret_list)-1]) - float(ret_list.adjClose[0]))/float(ret_list.adjClose[0])
        except:
            ret = 'None'
        return ret

if __name__ == '__main__':
    s = SecCrawler()
    s.parse_form4('2015-12', '2016-11')

