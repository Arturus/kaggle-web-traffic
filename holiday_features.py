#Define  few functions to create holiday features from the time series
#For now, these are only intended to ork with DAILY sampled data

import pandas as pd
import numpy as np




def encode_all_holidays__daily(dates_range):
    """
    Encode all fixed and moving holidays, and corresponding holiday shoulders.
    Intended for daily sampled data only.
    """
    

    def get_fixed_date_holidays__daily(dates_range, month_day):
        """
        Get YYYY-mm-DD holidays,
        for holidays that occur yearly on fixed dates.

        For daily sampled data only.
        
        In USA:
        Christmas, New Year, 4th of July, Halloween, Cinco de Mayo
        Valentine's Day, Veteran's Day
        
        other international:
            ...
        """
#        return ['{}-{:02d}-{:02d}'.format(i.year,i.month,i.day) for i in dates_range if ((i.month==int(month_day[:2])) and (i.day==int(month_day[4:])))]
#        print([(i.month, i.day) for i in dates_range])
#        print([i for i in dates_range if ((i.month==int(month_day[:2])) and (i.day==int(month_day[4:])))])
        return [i.strftime('%Y-%m-%d') for i in dates_range if ((i.month==int(month_day[:2])) and (i.day==int(month_day[3:])))]

    # =============================================================================
    # MOVING holidays [variable date]
    # =============================================================================
    def get_thanksgivings__daily(dates_range):
        """
        Get Thanksgiving holiday dates within the few years time range
        """
    #    4th Thurs of Novmber...
    #    if (month==11) and (dayofweek=='Thurs') and (22<=dayofmonth<=28)
        thanksgiving_dates = []
        #...
        return thanksgiving_dates
    
    def get_Easters__daily(dates_range):
        """
        Get Easter holiday dates within the few years time range
        """
        easter_dates = []
        #...
        return easter_dates  
        
#    def encode_custom_dates__daily(dates_range,dates_list):
#        """
#        Encode custom days and optionally shoulder days.
#        For daily sampled data only.
#        
#        E.g. Superbowl Sunday
#        suberbowl_dates = ['2014-02-02','2015-02-01','2016-02-07','2017-02-05','2018-02-04','2019-02-03']
#        shoulders = [...]
#        """
#        return dates_range 
    
    def spiral_encoding(dates_range, holiday_date, shoulder):
        """
        Encode holiday and shoulders as a spiral:
        Rotation over 2pi, with radius goes from 0 to 1 [on holiday] back to 0
        """
        N_real_days = len(dates_range)
        real_min = min(dates_range)
        real_max = max(dates_range)
        dates_range_padded = pd.date_range(real_min-shoulder-2, real_max+shoulder+2, freq='D')
#        print(dates_range)
#        print(dates_range_padded)
        
        df = pd.DataFrame()
        df['date'] = dates_range_padded.values 
        Ndays = len(df)
        
#        print(holiday_date)
        _ = df.loc[df['date']==holiday_date]
        if len(_)>0:
            ind = _.index.values[0]
        #If this holiday is completely out of bounds of the time series input,
        #ignore it [assumed additive holiday effects, so just add 0's]
        else:
            return np.zeros((N_real_days,2))
        
        #For radius: triangle kernel centered on holiday
        r = np.zeros(Ndays)
        r[ind-shoulder-1:ind+1] = np.linspace(0.,1.,shoulder+2)
        r[ind:ind+shoulder+2] = np.linspace(1.,0.,shoulder+2)

        #For anlge: go from phase [0,pi], with holiday at pi/2
        theta = np.zeros(Ndays)
        theta[ind-shoulder-1:ind+shoulder+2] = np.linspace(0., np.pi, 2*shoulder+3)
        #Convert to Cartesian:
        df['r'] = r 
        df['theta'] = theta
        df['x'] = df['r']*np.cos(df['theta'])
        df['y'] = df['r']*np.sin(df['theta'])
        v = df[((df['date']>=real_min) & (df['date']<=real_max))]
        v = v[['x','y']].values
#        print(v, v.sum(axis=0), v.sum(axis=1))
        return v
    
    
    
    Ndays = len(dates_range)
    
    #Fixed Holidays [add other international ones as needed]:
    xmas_dates = get_fixed_date_holidays__daily(dates_range, '12-25')
    new_years_dates = get_fixed_date_holidays__daily(dates_range, '01-01')
    july4_dates = get_fixed_date_holidays__daily(dates_range, '07-04')
    halloween_dates = get_fixed_date_holidays__daily(dates_range, '10-31')
    cincodemayo_dates = get_fixed_date_holidays__daily(dates_range, '05-05')
    valentines_dates = get_fixed_date_holidays__daily(dates_range, '02-14')
    veterans_dates = get_fixed_date_holidays__daily(dates_range, '11-11')
    #taxday_dates = get_fixed_date_holidays__daily(dates_range, '04-15')
    

    #Rule Based Moving Holidays
    thanksgiving_dates = get_thanksgivings__daily(dates_range)
    easter_dates = get_Easters__daily(dates_range)
    #... Labor Day, Memorial Day, President's Day, MLK Day, Columbus Day, Tax Day
    #Custom / Single Event moving Holidays
    suberbowl_dates = ['2014-02-02','2015-02-01','2016-02-07','2017-02-05','2018-02-04','2019-02-03']
    
    #Dict of holiday dates: shoulder halfwidth  [-S, -S+1, ..., holiday, holiday+1, ..., holiday+S]
    #for now just use 3 as the shoulder width for all "major" holidays, 0 or 1 for "minor" holidays
    #Use ODD numbers for shoulder sizes
    holidays = {'xmas_dates':(xmas_dates,3),
                'new_years_dates':(new_years_dates,3),
                'july4_dates':(july4_dates,1),
                'halloween_dates':(halloween_dates,1),
                'cincodemayo_dates':(cincodemayo_dates,1),
                'valentines_dates':(valentines_dates,1),
                'veterans_dates':(veterans_dates,1),
                'thanksgiving_dates':(thanksgiving_dates,3),
                'easter_dates':(easter_dates,1),
                'suberbowl_dates':(suberbowl_dates,1),
                }
#    print(holidays)
    
    
    #Assume additive holiday effects: (which should almost never matter anyway 
    #for small shoulders unless there is overlap beteen some holidays. E.g. with shoulder=3, 
    #Christmas and New Year's do NOT overlap.)
#    encoded_holidays = pd.DataFrame()
#    encoded_holidays['date'] = dates_range.values
    _ = np.zeros((Ndays,2))
    #Iterate through each holiday, accumulating the effect:
    for mmm in holidays.values():
        shoulder = mmm[1]
        #Since date series is potentially over few years, could have e.g. several Christmas furing that time range
        for hd in mmm[0]:
            _ += spiral_encoding(dates_range, hd, shoulder)
#    print(_)
    return _