'''
TODO

DONE 
Log this - nullState -> idleState: Startup
Log cat flap lock-unlock messages
'''

import time
import argparse
import re

from datetime import datetime, timedelta
from os import path
import sys, os

# from base_logger import logger


def process_line(l:str):
    match = re.search("(\d\d\d\d-\d\d-\d\d) (\d\d:\d\d:\d\d,\d\d\d) (\w+)\((\d+)\):(\w+?) (.+)", l)
    ldate, ltime, lfname, lline, llevel, lmessage = (match.group(1), match.group(2), match.group(3), match.group(4), match.group(5), match.group(6))  #
    ltimestamp = datetime.strptime(f"{ldate} {ltime}", '%Y-%m-%d %H:%M:%S,%f')
    return ltimestamp, lmessage


def timestring(dtime: datetime) -> str:
    retval = ""

    # print(f"days {dtime.days}, secs {dtime.seconds}, us {dtime.microseconds}")

    if dtime.days > 0:
        retval = f"{dtime.days} days"
    elif dtime.seconds > 60*60:
        retval = f"About {dtime.seconds/(60*60):.1f} hours"
    elif dtime.seconds > 40*60:
        retval = f"About 45 minutes"
    elif dtime.seconds > 25*60:
        retval = f"About half an hour"
    elif dtime.seconds > 12*60:
        retval = f"About 15 minutes"
    elif dtime.seconds > 8*60:
        retval = f"About 10 minutes"
    elif dtime.seconds > 4*60:
        retval = f"About 5 minutes"
    elif dtime.seconds > 2*60:
        retval = f"A few minutes"
    elif dtime.seconds > 50:
        retval = f"About a minute"
    elif dtime.microseconds > 25*1e6:
        retval = f"About 30 seconds"
    elif dtime.microseconds > 13*1e6:
        retval = f"About 15 seconds"
    elif dtime.microseconds > 8*1e6:
        retval = f"About 10 seconds"
    elif dtime.microseconds > 3*1e6:
        retval = f"About 5 seconds"
    elif dtime.microseconds > 0.8*1e6:
        retval = f"About a second"
    else:
        retval = f""
    return retval

def process_log(inputfile: str, outputfile: str) -> bool:
    print(f"Processing {inputfile}")


    try:
        out = open(outputfile, 'w')
        out.write("@startuml\n")
        out.write("participant nullState\n")
        out.write("participant idleState\n")
        out.write("participant unlockedState\n")
        out.write("participant movementLockedState\n")
        out.write("participant mouseLockedState\n")
        out.write("participant flapControl\n")

        with open(inputfile, 'r') as f:
            # 2023-11-13 08:43:27,787 states(131):INFO PUML newpage
            lprevious = f.readline()
            ts, _ = process_line(lprevious)
            out.write(f"note left of nullState: {ts:%Y-%m-%d %H:%M:%S}\n\n")

            for l in f:
                if re.search("INFO PUML", l) == None:
                    continue
                tprevious, _ = process_line(lprevious)
                t, msg = process_line(l)
                dt = t - tprevious
                lprevious = l

                if dt.microseconds > 100000:
                    timestr = timestring(dt)
                    # Pull the states from and to so we can determine the print format
                    match = re.search("PUML (\w+) -+> (\w+): (.+)", msg)
                    fromstate, tostate, message = (match.group(1), match.group(2), match.group(3))
                    # Print the message
                    if timestr != "":
                        if fromstate == "idleState" and tostate != "idleState":
                            # Leaving idle gets formatted as a separator
                            out.write(f"== {timestr} later ==\n")
                            out.write(f"note left of nullState: {t:%Y-%m-%d %H:%M:%S}\n\n")
                        else:
                            # Other messages as notes
                            out.write(f"note over {fromstate}: {timestr} later\n")
                out.write(msg.lstrip("PUML ") + "\n")
        out.write("@enduml\n")
    except Exception as e:
        print(f"Crash - \n{e}")
    finally:
        out.close()

    return False



def main(): 
    parser = argparse.ArgumentParser(description="PUML - Create a Plant UML representation from a log")
    parser.add_argument('--inputfile', 
        help="The log to read",
        default='./log/catflap.log',
        required=False,
        type=str, action='store',)
    parser.add_argument('--outputfile', 
        help="The PUML file to write",
        default='./log/catflap.puml',
        required=False,
        type=str, action='store',)
    
    args = parser.parse_args()
    if os.path.isfile(args.inputfile) == True:
        process_log(args.inputfile, args.outputfile)

  

if __name__ == "__main__":
    main()