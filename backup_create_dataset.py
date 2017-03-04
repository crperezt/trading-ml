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

        dataset_file = open('dataset_spy_12.csv', 'a+')
        dataset_file.write('COMPANY,DATE,NBC1,NBC2,NBC3,NBC4,NBC5,NBC6,NBC7,NBC8,NBC9,NBC10,NBC11,NBC12,NBV1,NBV2,NBV3,NBV4,NBV5,NBV6,NBV7,NBV8,NBV9,NBV10,NBV11,NBV12,SECTOR,MKTCAP,RET1,RET6,RET12\n')
        
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

                
                dataset_file.write(c + ',' + sorted_cd[k][0] + ',')
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
                    ret_list = self.get_return_13(c, sorted_cd[j][0])
                    dataset_file.write(str(csector) + ',' + str(cmktcap) + ',' + str(ret_list[0]) + ',' + str(ret_list[1]) + ',' + str(ret_list[2]) + '\n')
                else:
                    dataset_file.write(str(csector) + ',' + str(cmktcap) + '\n')

        dataset_file.close()
            


    def get_return_13(self, company, month):
        #month is 'YYYY-MM' format

        month_int = int(month[5:7])
        month_first = month + '-01'

        last_days = {1: '31', 2: '28', 3: '31', 4: '30', 5: '31', 6: '30',
        7:'31', 8:'31', 9:'30', 10:'31', 11:'30', 12:'31'}

        month_last = month + '-' + last_days[month_int]
        month6_int = month_int + 5
        if month6_int > 12:
            month6_int = month6_int - 12
            year6_int = int(month[0:4]) + 1

        month12_int = month_int + 11
        year12_int = int(month[0:4])
        if month12_int > 12:
            month12_int = month_int - 12
            year12_int = int(month[0:4]) + 1
            
        month6_str = str(month6_int) if month6_int > 9 else '0' + str(month_6int)
        month12_str = str(month12_int) if month12_int > 9 else '0' + str(month_12int)

        month_last_6 = str(year6_int) + '-' + month6_str + '-' + last_days[month6_int]
        month_last_12 = str(year12_int) + '-' + month12_str + '-' + last_days[month12_int]

        print "Month of return: " + month + '\n'
        print "First of month: " + month_first + '\n'
        print "Last of month: " + month_last + '\n'
        print "Last of 6 months: " + month_last_6 + '\n'
        print "Last of 12 months: " + month_last_12 + '\n'


        price_list = []
        ret_list = []
        for attempts in range (20):
            try:
                ret_list[0] = pd.read_json("https://api.tiingo.com/tiingo/daily/" + company.lower() + "/prices?startDate=" + month_first + "&endDate=" + month_last + "&token=8a387055f2f4081b89abfc6b3044284e958f178e")
                ret_list[1] = pd.read_json("https://api.tiingo.com/tiingo/daily/" + company.lower() + "/prices?startDate=" + month_first + "&endDate=" + month6_last + "&token=8a387055f2f4081b89abfc6b3044284e958f178e")
                ret_list[2] = pd.read_json("https://api.tiingo.com/tiingo/daily/" + company.lower() + "/prices?startDate=" + month_first + "&endDate=" + month12_last + "&token=8a387055f2f4081b89abfc6b3044284e958f178e")
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

        try:
            #ret = (float(ret_list.adjClose[len(ret_list)-1]) - float(ret_list.adjClose[0]))/float(ret_list.adjClose[0])
            ret_list[0] = (float(price_list[0].adjClose[len(price_list[0])-1]) - float(price_list[0].adjClose[0]))/float(price_list[0].adjClose[0])
            ret_list[1] = (float(price_list[1].adjClose[len(price_list[1])-1]) - float(price_list[1].adjClose[0]))/float(price_list[1].adjClose[0])
            ret_list[2] = (float(price_list[2].adjClose[len(price_list[2])-1]) - float(price_list[2].adjClose[0]))/float(price_list[2].adjClose[0])
        except:
            ret_list[0] = 'None'
            ret_list[1] = 'None'
            ret_list[2] = 'None'
        return ret_list

if __name__ == '__main__':
    s = SecCrawler()
    s.parse_form4('2010-01', '2016-12', returns=True)

