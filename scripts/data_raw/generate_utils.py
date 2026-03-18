import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import requests  # for UniProd web API
from pprint import pprint
import pickle  # to save some query that take long to execute
from multiprocessing import Pool, Lock  # for multithreading
import threading
import time
from Bio import SeqIO  # to read fasta file
from pathlib import Path
from filelock import FileLock
from scripts.initialize import *

# get bressin stuff
from bressin_negativePfamTerms import PfamTerms as BressinPfamTerms
from bressin_negativeSetGOterms import GOterms as BressinGOterms

nonNegative_GOterms = BressinGOterms.GO_term[BressinGOterms.GO_term.notnull()].unique()
nonNegative_IPRterms = BressinPfamTerms.IPR[BressinPfamTerms.IPR.notnull()].unique()


# Parameters
APIroot_InterPro = "https://www.ebi.ac.uk:443/interpro/api/"
APIroot_MobiDB = "https://mobidb.org/api/"


# represents a file that has request answers store
# trades network load against file system load (which is normally a speedup)
class Cache:
    def __init__(self, path) -> None:  # needs to be created outside of an thread
        self.folderPath = path
        self.content = {}
        self.existingKeys = set()

        # create folder if not existent
        if self.folderPath.exists() == False:
            self.folderPath.mkdir(parents=True)
        else:  # if folder existent read contents
            #self.existingKeys.extend(list(self.folderPath.iterdir()))
            self.existingKeys = {p.name for p in self.folderPath.iterdir()}

    # updates the cache on the hard drive
    def save_old(self):
        existingKeyFiles = [x.stem for x in self.folderPath.iterdir()]
        # log(f"Saving {self.path} with {len(self.content.keys())} keys to folder where {len(existingKeyFiles)} files are present")
        for key in self.content.keys():
            if not key in existingKeyFiles:  # not already in saved cache
                filePath = self.folderPath.joinpath(key)
                # log(f"Writing {filePath}")
                with filePath.open("w+") as f:
                    # pickle.dump({key:value},f)
                    d = {key: self.content[key]}
                    # log(f"Write {key}: {d}")
                    f.write(json.dumps(d, indent=4, sort_keys=True))

    # writes key/value to file
    def save(self, key, value):
        filePath = self.folderPath.joinpath(key)
        # log(f"Writing {filePath}")
        with filePath.open("w+") as f:
            # pickle.dump({key:value},f)
            d = {key: value}
            #f.write(json.dumps(d, indent=4, sort_keys=True))
            f.write(json.dumps({key: value}, indent=4, sort_keys=True))
        self.existingKeys.add(key)

    # updates value in cache
    # gets cach update file and writes to files
    def update(self, d):
        for key in d.keys():
            key_cleaned = key.replace("/", "_").replace("\\", "_")  # replace slashes with underscore
            if key_cleaned not in self.existingKeys:
                self.save(key_cleaned, d[key])
                

    # get value from cache
    def get(self, key):
        key = key.replace("/", "_").replace("\\", "_")  # replace slashes with underscore
        if not key in self.existingKeys:  # not in cache
            return None
        else:
            filePath = self.folderPath.joinpath(key)
            with filePath.open("r") as f:
                lines = f.read()
                d = json.loads(lines)

                if type(d) == dict:
                    key = list(d.keys())[0]
                    self.content[key] = d[key]
                else:  # something went wrong. get the file lines
                    # log(f"{key}")
                    log(f"{filePath} lines: {d}")


# get suffix for taxon IDs
#  #get relevant suffix
def getSuffix(taxonID): #TODO: maybe make an interPro querey for the tason ids? maybe we get an suffix somwhere there?
    match taxonID:
        case 9606:
            return "_HUMAN"
        case 561:
            return "" #Escherichia, parent: e.g. "ECO24", "ECOL6"
        case 590:
            return "" #Salmonella, parent: e.g. "SALPA", "SALCH", "SALTY"
        case 3702:
            return "_ARATH" #Arabidopsis thaliana, 
        case 212608:
            return "" #Chrysis elegans, parent to Chrysis elegans elegans (TODO)
        case 7215:
            return "" #Drosophila, parent class, has also 32280 (drosophila) or 32341 (Sophophora) (TODO) best guess is "_DROME" but idk if there are also others!
        case 10090:
            return "_MOUSE" #mus musculus
        case 559292:
            return "_YEAST" #bakers yeast
        
        case other:
            raise RuntimeError(f"No suffix defined for taxon ID {taxonID}")


## InterPro-Annotations ##
IPR_Annotation_Cache = Cache(CACHE.joinpath("IPR_Annotations/"))


def getAnnotations_InterPro(IPR_Annotation_Accession, IPR_Annotation_Cache):
    global APIroot_InterPro
    APIroot = APIroot_InterPro

    # get specific domain name and Go annotation from InterPro
    output = IPR_Annotation_Cache.get(IPR_Annotation_Accession)
    if output == None:  # no entry found in cache
        url = APIroot + f"entry/interpro/{IPR_Annotation_Accession}"  # gets us the domain specific information
        ret = requests.get(url)
        if ret.status_code == 200:
            IPR_info = json.loads(ret.text)
            # plog(d)
        else:
            log(f"getAnnotations_InterPro({IPR_Annotation_Accession}): WARNING: HTTP ERROR: {ret}")

        # get domain type (binding or not)
        go_terms = IPR_info["metadata"]["go_terms"]
        if go_terms != None:
            is_binding = "GO:0003723" in [terms["identifier"] for terms in go_terms]
            ty = 1 if is_binding else 0
        else:  # if there is not go annotation, the domains is not RNA binding
            ty = 0

        # get domain name
        sName = IPR_info["metadata"]["name"]["short"]

        # get familiy name
        familyAccession = IPR_info["metadata"]["hierarchy"]["accession"]
        url = APIroot + f"entry/interpro/{familyAccession}"  # gets us the family domain specific informations
        ret = requests.get(url)
        if ret.status_code == 200:
            IPR_info_family = json.loads(ret.text)
            # plog(d)
        else:
            log(
                f"getAnnotations_InterPro({IPR_Annotation_Accession}): WARNING: Getting name for family {familyAccession}\t HTTP ERROR: {ret}"
            )
        name = IPR_info_family["metadata"]["name"]["short"]
        fam_ty = IPR_info_family["metadata"]["type"]
        if fam_ty != ty:
            RuntimeError(
                "For accession {IPR_Acession} ({sName}), ty is {ty}, but for family accession {familyAccession} ({name}), ty is {fam_ty}"
            )

        output = (ty, name, sName)
    elif type(output) == int:  # an error code
        return output
    return output


## InterPro-Protein ##
IPR_Protein_Cache = Cache(CACHE.joinpath("IPR_Proteins/"))
IPR_ProteinByName_Cache = Cache(CACHE.joinpath("IPR_ProteinsByName/"))

def getProtein_InterPro(ID, IPR_Annotation_Cache, IPR_Protein_Cache, IPR_ProteinByName_Cache, byName=False):
    global APIroot_InterPro
    APIroot = APIroot_InterPro

    IPR_Annotation_Cache_update = {}

    ## Get Protein data
    if(byName):
        output = IPR_ProteinByName_Cache.get(ID)
    else:
        output = IPR_Protein_Cache.get(ID)

    if output == None:  # not in cache
        # url = root+f"protein/reviewed/{accession}"
        url = APIroot + f"protein/uniprot/{ID}/entry/interpro" # gets us the domains and the sequence
        ret = requests.get(url)
        if ret.status_code == 200:
            d = json.loads(ret.text)
        else:
            return ret.status_code
       
        # Get annotations via entries_url (InterPro now returns a summary + link)
        annotations = []
        accessions_seen = set()

        entries_url = d.get("entries_url")
        if not entries_url:
            # fallback: combined endpoint mentioned in docs
            entries_url = APIroot + f"entry/interpro/protein/uniprot/{ID}"

        # ensure reasonable page size; follow pagination via "next"
        if "page_size=" not in entries_url:
            sep = "&" if "?" in entries_url else "?"
            entries_url = f"{entries_url}{sep}page_size=200"

        next_url = entries_url
        while next_url:
            r = requests.get(next_url, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"InterPro entries fetch failed: {r.status_code} {next_url}")
            j = r.json()
            for item in j.get("results", []):
                md = item.get("metadata", {})
                acc = md.get("accession")
                if md.get("type") != "domain" or not acc:
                    continue
                # keep your existing per-accession annotation typing
                ty, name, sName = getAnnotations_InterPro(acc, IPR_Annotation_Cache)
                IPR_Annotation_Cache_update[acc] = (ty, name, sName)
                proteins = item.get("proteins", [])
                if not proteins:
                    continue
                for loc in proteins[0].get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        fr = frag.get("start"); to = frag.get("end")
                        if fr is not None and to is not None:
                            annotations.append((fr, to, ty, name, sName))
                accessions_seen.add(acc)
            next_url = j.get("next")
            if next_url is None:
                break

        # Check negatives using collected accessions + metadata.go_terms
        bressinPossibleNegative = True
        if accessions_seen.intersection(set(nonNegative_IPRterms)):
            bressinPossibleNegative = False
        go_terms = (d.get("metadata") or {}).get("go_terms") or []
        for gt in go_terms:
            if isinstance(gt, dict) and gt.get("identifier") in set(nonNegative_GOterms):
                bressinPossibleNegative = False
                break

        # extract other data
        # RBP_Name = d["metadata"]["gene"] #e.g. PUM2
        uniprodID = d["metadata"]["accession"]  # e.g. Q96RI9
        RBP_Name = d["metadata"]["gene"]  # e.g. PUM2
        seq = d["metadata"]["sequence"]  # AA sequence
        taxID = int(d["metadata"]["source_organism"]["taxId"])

        output = (uniprodID, RBP_Name, taxID, seq, annotations, bressinPossibleNegative)
    elif type(output) == int:  # an error code
        # log("loaded error code, no need to refresh")
        return output

    return output, IPR_Annotation_Cache_update


## MOBI-DB-Lite ##

MobiDB_Cache = Cache(CACHE.joinpath("MobiDB/"))


def getIDR_MobiDB(uniprodID, MobiDB_Cache,taxonID):
    global APIroot_MobiDB
    APIroot = APIroot_MobiDB

    output = MobiDB_Cache.get(uniprodID)
    if output == None:  # not found in cache
        url = APIroot + f"download?format=json&projection=prediction-disorder-mobidb_lite&acc={uniprodID}&ncbi_taxon_id={taxonID}"
        ret = None
        while ret == None:
            try:
                ret = requests.get(url)
            except requests.exceptions.ConnectTimeout as e:
                log(f"MobiDB_getIDR({uniprodID}): WARNING: API ERROR -> trying again after 5 seconds.")
                ret = None
                time.sleep(5)  # np.random.rand()*5) #sleep between 0 and X seconds
                # -> should reduce server load and therfore increase response rate

        if ret.status_code == 200:
            if ret.text == "":
                # this means normally that MobiDB wants to say:
                # "ERROR: this UniProt accession is not available in the database!""
                # log(f"{uniprodID}: JSON DECODE ERROR ({e}) from ret.text: \"{ret.text}\"")
                output = ()
            else:
                d = json.loads(ret.text)

                keys = list(d.keys())
                if "prediction-disorder-mobidb_lite" in keys:
                    output = d["prediction-disorder-mobidb_lite"]["regions"]
                else:  # there are other disorder entries but non from mobidb-light itself
                    if keys != []:  # if there are other keys
                        log(f"MobiDB_getIDR({uniprodID}): WARNING: no 'prediction-disorder-mobidb_lite' entry found (only: {keys})")
                    else:
                        pass  # no entries at all.
                    output = ()

        else:
            return ret.status_code
    elif type(output) == int:  # an error code
        return output
    return output
