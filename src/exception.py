import sys
import logging

def error_massage_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_massage="Error occured in python script name [{0}] line number [{1}] error massage[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_massage

    

class CustomExecption(Exception):
    def __init__(self, error_massage,error_detail:sys):

        super().__init__(error_massage)
        self.error_massage=error_massage_detail(error_massage,error_detail=error_detail)

        def __str__(self):
            return self.error_massage 
           