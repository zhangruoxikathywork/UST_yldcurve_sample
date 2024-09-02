##########################################
# This is the wrapper program for        #
# cleaning data into parms, curve        #
# calculations with and without          #
# taxbility, produce predicted vs actual #
# price and yield, and plot curves       #
##########################################


import sys
import os
from datetime import date, datetime
import pandas as pd
import numpy as np
from tabulate import tabulate
from fpdf import FPDF
from PyPDF2 import PdfReader, PdfWriter
import textwrap
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
# os.chdir("curve_utils/src/development") # KZ: for my vscode directory bug...


def create_latex_with_images(output_dir, date_folder, plot_folder, output_latex_file):
    
    image_directory = os.path.join(output_dir, date_folder, plot_folder)
    output_latex_path = os.path.join(output_dir, output_latex_file)
    
    # LaTeX document header and setup
    header = r'''\documentclass{article}
    \usepackage{graphicx}
    \usepackage[margin=1in]{geometry} % Adjust page margins if necessary
    \begin{document}
    '''

    # LaTeX document footer
    footer = r'''\end{document}'''

    # Start writing the .tex file
    with open(output_latex_path, 'w') as file:
        file.write(header)
        
        # Loop through all .png files in the specified directory
        for image in sorted(os.listdir(image_directory)):
            if image.endswith('.png'):
                image_path = os.path.join(date_folder, plot_folder, image).replace('\\', '/')
                # LaTeX code to include an image
                file.write(r'\begin{figure}[h!]\centering' + '\n')
                file.write(r'\includegraphics[width=\textwidth]{' + image_path + '}\n')
                file.write(r'\caption{' + image[:-4].replace('_', r'\_') + '}\n')
                file.write(r'\end{figure}' + '\n')
        
        file.write(footer)

# Example usage
# create_latex_with_images('output_dir', 'date_folder', 'plot_folder', 'output_latex_file.tex')




#%% Write to txt and pdf

def write_df_to_txt(df, output_dir, estfile, esttype, ctype=False):

    if ctype is not False:
        df = df.xs(ctype, level='ctype')
    # df = df.reset_index(drop=True)
    df = df.reset_index()
    if 'quotedate_ymd' in df.columns:
        df.rename(columns={'quotedate_ymd': 'quodatedate'}, inplace=True)
    df['quotedate'] = df['quotedate'].astype(int).astype(str)
    df['quotedate'] = pd.to_datetime(df['quotedate'], format='%Y%m%d')
    df['quotedate'] = df['quotedate'].dt.strftime('%m/%d/%Y')
    df.rename(columns={'quotedate': 'Quote Date'}, inplace=True)

        # df_fixed.set_index(['quotedate'], inplace=True, drop=True)

    def format_numeric_to_n_decimals(x):
        """Round numeric entries to three decimal places."""
        if isinstance(x, (int, float)):
            return round(x, 2)  # round(100*x, 2)
        else: 
            return x
    
    df = df.map(format_numeric_to_n_decimals)
    formatted_table = tabulate(df, headers='keys', tablefmt='plain', numalign="right", stralign="right", showindex=False)

    if ctype is not False:
        with open(f'{output_dir}/{estfile}_{esttype}_{ctype}.txt', 'w') as f:
            f.write(formatted_table)
    else:
        with open(f'{output_dir}/{estfile}_{esttype}.txt', 'w') as f:
            f.write(formatted_table)


def write_txt_to_pdf(output_dir, estfile, esttype, ctype=False, tax=False, padj=None, date=None):

    filename = f'{estfile}_{esttype}'
    if ctype is not False:
        filename += f'_{ctype}'
    input_path = f"{output_dir}/{filename}.txt"
    output_path = f"{output_dir}/{filename}.pdf"

    # Map title names
    esttype_list = ['parbd_rate', 'parbd_cprice', 'zerobd_rate', 'zerobd_cprice',
                    'annuity_rate', 'annuity_cprice', 'curve', 'reture_df', 'total_ret',
                    'yld_ret', 'yld_excess', 'short', 'predyld']

    title_name = ["ESTIMATED PAR BOND YIELDS",    
                  "ESTIMATED PAR BOND CLEAN PRICES",    
                  "ESTIMATED ZERO BOND YIELDS",    
                  "ESTIMATED ZERO BOND CLEAN PRICES",    
                  "ESTIMATED ANNUITY YIELDS",    
                  "ESTIMATED ANNUITY CLEAN PRICES",    
                  "ESTIMATED FORWARD RATES",    
                  "ESTIMATED MONTHLY RETURN",    
                  "ESTIMATED MONTHLY TOTAL RETURN",    
                  "ESTIMATED MONTHLY YIELD RETURN",    
                  "ESTIMATED MONTHLY RETURN IN EXCESS OF YIELD",    
                  "SHORT",    
                  "PREDICTED PRICE AND YIELD"]

    mapping = {key: value for key, value in zip(esttype_list, title_name)}
    title = mapping[esttype]

    # Setting up the PDF document in landscape format
    c = canvas.Canvas(output_path, pagesize=landscape(letter))
    width, height = landscape(letter)
    margin = 50
    page_number = 1

    # if date is None:
    #     date = datetime.today()
    #     date = date.strftime('%d-%b-%y').upper()
    #     if date.startswith('0'):
    #         date = date[1:]
    # else:
    #     date = datetime.strptime(str(date), '%Y%m%d')
    #     date = date.strftime('%-d-%b-%y').upper()

    if date is None:
        date = datetime.today().strftime('%d-%b-%y').upper()  # Format date as 'DD-Mon-YY'
    else:
        date = datetime.strptime(str(date), '%Y%m%d').strftime('%d-%b-%y').upper()


    def add_page(column_headers):
        nonlocal page_number
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, height - margin + 15, f"{title} (in %)")
        # Map tax names
        tax_list = [False, 0, 1, 2, 3]
        tax_name = ["Not Taxable", "Not Taxable", "Fully Taxable", 
                    "Partially Tax-Exempt", "Fully Tax-Exempt"]
        tax_mapping = {key: value for key, value in zip(tax_list, tax_name)}
        tax_name = tax_mapping[tax]
        
        c.setFont("Helvetica-Oblique", 12)
        if padj is not None:
            c.drawString(margin, height - margin, f"{tax_name}, Curve Type: {ctype}, Principal Adj: {padj * 100:.0f}%, {date}")
        else:
            c.drawString(margin, height - margin, f"{tax_name}, Curve Type: {ctype}, {date}")

        c.setFont("Courier", 7.5)
        c.drawString(width - margin, margin / 2, f"Page {page_number}")
        # Draw column headers
        c.drawString(margin, height - margin - 35, column_headers)
        page_number += 1

    with open(input_path, 'r') as file:
        lines = file.readlines()
        column_headers = lines[0].strip()

    y_position = height - margin - 60  # Starting position below headers
    add_page(column_headers)  # Start first page

    for line in lines[1:]:  # Skip the header line
        if y_position < margin + 40:
            c.showPage()
            add_page(column_headers)
            y_position = height - margin - 70  # Reset y-position
        c.drawString(margin, y_position, line.strip())
        y_position -= 12  # Line spacing

    c.save()


def combine_pdfs(output_dir, folder_name):

    folder_path = os.path.join(output_dir, folder_name)
    
    # List of pdf files
    pdf_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.pdf')]
    pdf_writer = PdfWriter()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        pdf_reader = PdfReader(pdf_path)

        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])

    output_pdf_path = os.path.join(output_dir, folder_name + '.pdf')
    
    with open(output_pdf_path, 'wb') as out_pdf_file:
        pdf_writer.write(out_pdf_file)


def combine_csvs(output_dir, folder_name):
    folder_path = os.path.join(output_dir, folder_name)
    
    # List of csv files
    csv_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.csv')]

    dataframes = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    output_csv_path = os.path.join(output_dir, folder_name + '.csv')

    combined_df.to_csv(output_csv_path, index=False)