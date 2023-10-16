### This is the SwissParam webserver ligand force field parametrization code taken from:
### https://github.com/aaayushg/charmm_param/blob/master/swiss.py
### This code was originally written by Aayush Gupta.

from bs4 import BeautifulSoup
from xml.dom import minidom
import mechanize
import ssl
#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("-c", "--conf")
#args = parser.parse_args()

def swiss_func(filename):

    # sanity check
    lig_mol2 = filename.split('/')[-1]
    assert lig_mol2.endswith('.mol2')
    
    # initialize the browser.
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context
    br = mechanize.Browser()
    br.set_handle_robots(False)   # ignore robots
    br.set_handle_refresh(False)  # can sometimes hang without this
    br.addheaders = [('User-agent', 'Firefox.')]
    br.set_handle_redirect(mechanize.HTTPRedirectHandler)

    # login fill-in form and submit
    url = "http://old.swissparam.ch/"
    response = br.open(url)
    br.form = list(br.forms())[0]

    # upload the file to parametrize, parse xml output
    br.form = list(br.forms())[0]
    br.form.add_file(open(filename), 'text/plain', filename)
    response = br.submit()
    xml = response.read().strip()
    #print xml
    soup = BeautifulSoup(xml,'html.parser')
    for link in soup.find_all('a'):
        if 'swissparam' in link.get('href'):
            out_link = link.get('href')

    lig_zip = lig_mol2.replace('.mol2','.zip')
    out_link = out_link.replace('index.html',lig_zip)
    return out_link
    
    #Save links into a folder and download all links using wget
    #keep buffer time in wget (for job to complete on webserver)
