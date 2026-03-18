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
import time
from Bio import SeqIO  # to read fasta file

# get bressin stuff
from bressin_negativePfamTerms import PfamTerms as BressinPfamTerms
from bressin_negativeSetGOterms import GOterms as BressinGOterms

nonNegative_GOterms = BressinGOterms.GO_term[BressinGOterms.GO_term.notnull()].unique()
nonNegative_IPRterms = BressinPfamTerms.IPR[BressinPfamTerms.IPR.notnull()].unique()


# goes through all go terms and check if the given identifier exits
def checkGo(goTerms, target="GO:0003723"):
    if goTerms == None:
        return False
    for term in goTerms:
        if term["identifier"] == target:
            return True
    return False


##### CISBP-RNA #####


# parse binding domains from http://cisbp-rna.ccbr.utoronto.ca/index.php (which is from Ray 2013)
# DEPRECIATED! does not follow the new interface (InterPro uses)
def getCisBP(filePath, CisBPgtFilePath, forceRefresh=False, species_id=9606, silent=True):  # human 9606
    if forceRefresh or (os.path.isfile(filePath) == False):
        if not silent:
            print(f"Read CisBP-RNA file from {CisBPgtFilePath}")
        d_orig = pd.read_csv(CisBPgtFilePath, sep="\t")

        # reformat data
        if not silent:
            print(f"Reformat data from CIsBP-RNA file")
        Species = []  # Species (e.g. "Homo_sapiens")
        RBP_Names = []  # RBP name
        Gene_IDs = []  # Gene ID
        Protein_IDs = []  # Protein ID (will be the filename in the embeedings folder)
        Protein_Sequences = []  # AA sequence
        Domains = []  # triplet: (from, to, type)
        Canonical = []

        # get data from CisRB-RNA file
        for i in d_orig.index:
            temp = d_orig["Protein_seq"][i]
            if temp in Protein_Sequences:  # for some reason, some sequences are included multiple times. Do not add them
                continue

            Protein_Sequences.append(temp)
            Species.append(d_orig["Species"][i])
            RBP_Names.append(d_orig["RBP_Name"][i])
            Gene_IDs.append(d_orig["Gene_ID"][i])
            Protein_IDs.append(d_orig["Protein_ID"][i])
            Canonical.append(None)  # unknown yet

            types = d_orig["Pfam_DBDs"][i].split(",")
            froms = d_orig["Pfam_froms"][i].split(",")
            tos = d_orig["Pfam_tos"][i].split(",")

            prot_domains = []
            for fr, to, name in zip(froms, tos, types):
                if name == "UNKNOWN":
                    continue

                # TODO: hieracy of names? are the CisBP-RNA names specific or familiy wide?!?

                prot_domains.append((int(fr), int(to), 1, name, name))  # CisBp-RNA only contains RNA-binding domain annotations!

            Domains.append(prot_domains)

        # create new dataframe
        RBPs = pd.DataFrame(
            data={
                "Species": Species,
                "RBP_Name": RBP_Names,
                "Gene_ID": Gene_IDs,
                "Protein_ID": Protein_IDs,
                "Protein_seq": Protein_Sequences,
                "domains": Domains,
                "canonical": Canonical,
            }
        )

        # Find canonical forms and add them
        if not silent:
            print("Get canonical flag based on UniProt reviewd Entries")

        num_rbp = len(RBPs.RBP_Name.unique())
        for i, RBP_Name in enumerate(tqdm(RBPs.RBP_Name.unique())):
            # RBP_Name e.g. "FUS"

            if not silent:
                print(f"{i}/{num_rbp}({(i/num_rbp)*100:0.2f}%)\t{RBP_Name}", end="")

            # rempace gene with accession if you want to search with P35637 instead of FUS_HUMAN
            query = f"organism_id:{species_id}+AND+gene:{RBP_Name}+AND+reviewed:true"
            url = f"https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=json&includeIsoform:false&query={query}"

            ret_json = requests.get(url).text
            d = json.loads(ret_json)

            if not "results" in d.keys():
                print(f"HTML ERROR: {json.dumps(d, indent=4)}")

            else:
                q_results = len(d["results"])
                if not silent:
                    print(f"\tres={q_results}", end="")
                # print([r["uniProtkbId"]for r in d["results"]])

                if q_results > 0:
                    # find canonical form
                    isoforms = list(RBPs[RBPs.RBP_Name == RBP_Name].Protein_seq)

                    canonicalForm = None
                    for result in d["results"]:
                        if result["sequence"]["value"] in isoforms:
                            canonicalForm = result["sequence"]["value"]
                            break

                    if canonicalForm != None:
                        if not silent:
                            print("\tcanonical")
                        RBPs.loc[RBPs["Protein_seq"] == canonicalForm, ["canonical"]] = True
                    else:
                        if not silent:
                            print("\tno canoncial")

        if not silent:
            print(f"CisBP-RNA: Saving RBPs to {filePath}")
        RBPs.to_pickle(filePath)

    else:
        if not silent:
            print(f"CisBP-RNA: Loading RBPs from {filePath}")
        RBPs = pd.read_pickle(filePath)

    return RBPs


##### InterPro #####


# get all binding domain annotation listed in InterPro
def IP_get_RBDs(
    go_term_target="GO:0003723", ps=200, APIroot="https://www.ebi.ac.uk:443/interpro/api/", silent=True  # pagesize for the API call
):
    # get all InterPro DOMAINS with GO RNA binding annotation
    url = APIroot + f"entry/InterPro/?go_term={go_term_target}&type=domain&page_size={ps}"

    # get total
    ret = requests.get(url)
    if ret.status_code == 200:
        d = json.loads(ret.text)
        total_count = d["count"]

    # get data from webserver
    results = []
    pageCount = 0
    # print(f"Pages: ",end="")
    with tqdm(total=total_count) as pbar:
        while url != None:
            pageCount += 1
            ret = requests.get(url)
            if ret.status_code == 200:
                d = json.loads(ret.text)
                url = d["next"]
                results.extend(d["results"])
                # print(d)
                # print(f"{pageCount}/{int(d['count']/ps)+1} () ",end="")
                pbar.update(len(d["results"]))
            else:
                print(f"ERROR: {ret}")

    # make list of all binding domain accession
    bindingDomain_dict = {}  # key=accession, value=domain name

    for entry in tqdm(results):
        md = entry["metadata"]
        accession = md["accession"]
        name = md["name"]
        typ = md["type"]
        go_terms = md["go_terms"]

        # veryfy GO annotation
        gp_rna_binding = False
        for go_term in go_terms:
            if go_term["identifier"] == go_term_target:  # rna binding
                gp_rna_binding = True
                break
        if not gp_rna_binding:
            if not silent:
                print(f"INFO: {accession} ({name}) does not have GO annotation: {go_term_target}")
            continue

        # veryfiy that it is a domain
        type_id = "domain"
        if typ != type_id:
            if not silent:
                print(f"INFO: {accession} ({name}) is not a {type_id} but a {typ}!")
            continue

        # oterwise: hsould be fine and can be added
        bindingDomain_dict[accession] = {"name": name}

    if not silent:
        print(f"Binding domains found: {len(bindingDomain_dict)}:")

    return bindingDomain_dict


# get all (human, reviewed) RBP where at least one annotated RBD exists
# Runtime (circa): (Laptop=6min, PC=2.5min)
# returns list of InterPro protein entries
def IP_with_RBD(species_id=9606, ps=200, APIroot="https://www.ebi.ac.uk:443/interpro/api/", silent=True):  # pagesize for the API call
    bindingDomain_dict = IP_get_RBDs(APIroot=APIroot, silent=silent)  # get list of all binding domain accessions

    # for each domain type get all relevant protein entries (migth include duplicates!)
    results = []
    for domainType_id in tqdm(list(bindingDomain_dict.keys())):  # [12:13] is PUM
        domainType = bindingDomain_dict[domainType_id]
        domainName = domainType["name"]

        url = APIroot + f"protein/reviewed/entry/InterPro/{domainType_id}/taxonomy/uniprot/{species_id}"

        # get total
        ret = requests.get(url)
        if ret.status_code == 200:
            d = json.loads(ret.text)
            total_count = d["count"]
        elif ret.status_code == 204:
            total_count = 0

        # if not silent:
        #    print(f"Domain {domainType_id} ({domainName}) count = {total_count}")
        bindingDomain_dict[domainType_id]["count"] = total_count

        if total_count == 0:
            continue

        # get data from webserver
        # if not silent:
        #    print(f"\tGetting data from Webserver...")
        url = APIroot + f"protein/reviewed/entry/InterPro/{domainType_id}/taxonomy/uniprot/{species_id}/?page_size={ps}"

        domain_results = []
        pageCount = 0
        # print(f"Pages: ",end="")
        while url != None:
            pageCount += 1
            ret = requests.get(url)
            if ret.status_code == 200:
                d = json.loads(ret.text)
                url = d["next"]
                domain_results.extend(d["results"])
                # print(d)
                # print(f"{pageCount}/{int(d['count']/ps)+1} () ",end="")
            else:
                print(f"\t{domainType_id}\tHTTP ERROR @ {domainType_id}: {ret}")
                url = None

        results.extend(domain_results)

    # print Domain statistics:
    if not silent:
        print("Binding Domains:")
        sortIndices = np.argsort([d["count"] for d in bindingDomain_dict.values()])[::-1]
        for index in sortIndices:
            key = list(bindingDomain_dict.keys())[index]
            print(f"\t{key} c={bindingDomain_dict[key]['count']}  ({bindingDomain_dict[key]['name']})")

    return results


# gets all (human, reviewed) RBP whether with or without annotated RBP
# based on the web search and post-processed for GO annotation (RNA-binding) on molecule level
# ATTENTION: the molecule wide GO annotations seem to be too few and therfore datatset is smaller than expected
# (probably because most of molecule wide GO of RBP are simly missing in Inter Pro for some reason)
# returns list of InterPro protein entries
def IP_all_RBP(
    dataFolder,
    fromWeb=False,  # DON'T SET TRUE: for interPro the web query as well as the export do not respect the GO annotation
    # therfore the web query takes very long (20k samples). Its faster to download the export file from InterPro directly!
    ps=200,  # pagesize for the API call
    APIroot="https://www.ebi.ac.uk:443/interpro/api/",
    silent=True,
):
    if fromWeb:
        # url = APIroot+f"protein/UniProt/taxonomy/uniprot/9606/?go_term=GO:0003723?is_fragment=false" #207k
        # url = APIroot+f"entry/InterPro/protein/reviewed/taxonomy/uniprot/9606/?go_term=GO:0003723&is_fragment=false" #16k
        # url = APIroot+f"protein/reviewed/entry/InterPro/taxonomy/uniprot/9606/?search=&is_fragment=false" # -> 19422
        # url = APIroot+f"protein/reviewed/taxonomy/uniprot/9606/?go_term=GO%3A0003723&is_fragment=false&extra_fields=go_terms" # 20316
        # the above url is supposed to filtere by GO terms by it does not in reality.

        url = APIroot + f"protein/reviewed/taxonomy/uniprot/9606/?extra_fields=go_terms&page_size=200"
        # get total
        if not silent:
            print("Get total")
        ret = requests.get(url)
        if ret.status_code == 200:
            d = json.loads(ret.text)
            total_count = d["count"]
        if not silent:
            print(f"Total (unfiltered): {total_count}")

        # get data from webserver
        url += f"&page_size={ps}"
        if not silent:
            print("Get all data")
        results = []
        pageCount = 0
        # print(f"Pages: ",end="")
        with tqdm(total=total_count) as pbar:
            while url != None:
                pageCount += 1
                ret = requests.get(url)
                if ret.status_code == 200:
                    d = json.loads(ret.text)
                    url = d["next"]
                    results.extend(d["results"])
                    # print(d)
                    # print(f"{pageCount}/{int(d['count']/ps)+1} () ",end="")
                    pbar.update(len(d["results"]))
                else:
                    print(f"ERROR: {ret}")
    else:
        filePath = dataFolder + "interPro/export_9606_SwissProd_24023.json"

        if not silent:
            print(f"Loading file {filePath}")
        with open(filePath) as f:
            results_unfiltered = json.load(f)

        if not silent:
            print(f"Total (unfiltered): {len(results_unfiltered)}")

    # filter for molecule wide GO annotation
    # filtered = []

    # for result in results_unfiltered:
    #    go_terms=result["extra_fields"]["go_terms"]
    #    if go_terms != None:
    #        for go_term in go_terms:
    #            if go_term["identifier"] == "GO:0003723":
    #                filtered.append(result)
    #                break

    # if not silent:
    #    print(f"Total (fitlered): {len(filtered)}")

    return results_unfiltered


# collection of mapping: IPR domain infos so we do not have to make so many web API calls
IPR_infos = {}


def IPR_getDomainInfos(IPR_Acession, APIroot="https://www.ebi.ac.uk:443/interpro/api/"):
    global IPR_infos

    # get specific domain name and Go annotation from InterPro
    if not IPR_Acession in IPR_infos.keys():
        url = APIroot + f"entry/interpro/{IPR_Acession}"  # gets us the domain specific informations
        ret = requests.get(url)
        if ret.status_code == 200:
            IPR_info = json.loads(ret.text)
            # pprint(d)
        else:
            print(f"\t\tGetting name for {IPR_Acession}\t HTTP ERROR: {ret}")

        # get domain type (binding or not)
        go_terms = IPR_info["metadata"]["go_terms"]
        if go_terms != None:
            is_binding = "GO:0003723" in [terms["identifier"] for terms in go_terms]
            ty = 1 if is_binding else 0
        else:  # if there is not go annotaion, the domains is not RNA binding
            ty = 0

        # get domain name
        sName = IPR_info["metadata"]["name"]["short"]

        # get familiy name
        familyAccession = IPR_info["metadata"]["hierarchy"]["accession"]
        url = APIroot + f"entry/interpro/{familyAccession}"  # gets us the family domain specific informations
        ret = requests.get(url)
        if ret.status_code == 200:
            IPR_info_family = json.loads(ret.text)
            # pprint(d)
        else:
            print(f"\t\tGetting name for family {familyAccession}\t HTTP ERROR: {ret}")
        name = IPR_info_family["metadata"]["name"]["short"]
        fam_ty = IPR_info_family["metadata"]["type"]
        if fam_ty != ty:
            RuntimeError(
                "For acession {IPR_Acession} ({sName}), ty is {ty}, but for family acession {familyAccession} ({name}), ty is {fam_ty}"
            )

        IPR_infos[IPR_Acession] = ty, name, sName
        return ty, name, sName

    else:
        return IPR_infos[IPR_Acession]


MobiDB_infos = {}


def getMobiIDRs(uniProdAcession, APIroot="https://mobidb.org/api/"):
    global MobiDB_infos

    # get specific domain name and Go annotation from InterPro
    if not uniProdAcession in MobiDB_infos.keys():
        # url = APIroot+f"count?acc={uniProdAcession}"
        # ret = requests.get(url)
        # if(ret.status_code == 200):
        #    d = json.loads(ret.text)
        #    total_count = d["n"]
        # pprint(d)

        # url = APIroot+f"download?format=json&acc={uniProdAcession}"
        url = APIroot + f"download?format=json&projection=prediction-disorder-mobidb_lite&acc={uniProdAcession}&ncbi_taxon_id=9606"
        ret = None
        while ret == None:
            try:
                ret = requests.get(url)
            except requests.exceptions.ConnectTimeout as e:
                print(f"MobiDB-lite web API ERROR for {uniProdAcession}: {e}\n\ŧ-> trying again after 5 seconds.")
                ret = None
                time.sleep(5)  # np.random.rand()*5) #sleep between 0 and X seconds
                # -> should reduce server load and therfore increase response rate

        if ret.status_code == 200:
            if ret.text == "":
                # this means normally that MobiDB wants to say:
                # "ERROR: this UniProt accession is not available in the database!""
                # print(f"{uniProdAcession}: JSON DECODE ERROR ({e}) from ret.text: \"{ret.text}\"")
                r = []
                IPR_infos[uniProdAcession] = r
                return r
            else:
                d = json.loads(ret.text)

        if "prediction-disorder-mobidb_lite" in d.keys():
            r = d["prediction-disorder-mobidb_lite"]["regions"]
            IPR_infos[uniProdAcession] = r
            return r
        else:
            r = []
            IPR_infos[uniProdAcession] = r
            return r

    else:
        return IPR_infos[uniProdAcession]


def process_getInterPro(packed_parameters):
    global nonNegative_GOterms
    global nonNegative_IPRterms

    result, APIroot = packed_parameters

    md = result["metadata"]
    accession = md["accession"]  # e.g. Q8TB72 (for PUM2)

    Species = md["source_organism"]["taxId"]

    # get orientation (either positive set, negative set or None)
    bressinNegative = True  # True if any go term or IDP term indicates binding behavior
    bressinPositive = False  # True if molecule wide "GO:0003723" exists
    # check molecule wide GO annotations (requires file export from InterPro with extra filed "go_terms")
    if result["extra_fields"]["go_terms"] != None:
        for go_term in result["extra_fields"]["go_terms"]:
            if go_term["identifier"] in nonNegative_GOterms:
                bressinNegative = False
            if go_term["identifier"] == "GO:0003723":
                bressinPositive = True

    # get other informations
    # url = root+f"protein/reviewed/{accession}"
    url = APIroot + f"protein/uniprot/{accession}/entry/interpro"  # gets us the domains and the sequence
    ret = requests.get(url)

    if ret.status_code == 200:
        d = json.loads(ret.text)
        # pprint(d)
    elif ret.status_code == 204:  # no data so leave loop
        return accession, Species, None, None, None, [], None, None, None
    # elif ret.status_code == 408:
    #    time.sleep(61)
    #    d = json.loads(ret.text)
    else:
        print(f"\t\tGetting domain for {accession}\t HTTP ERROR: {ret}")
        return accession, Species, None, None, None, [], None, None, None

    # Check if the sample is in positive or negative set (according to bressin)
    if d["entry_subset"] != None and bressinNegative == None:
        for IPR in d["entry_subset"]:
            if IPR["accession"] in nonNegative_IPRterms:
                bressinNegative = False
                break
    if d["metadata"]["go_terms"] != None:  # and bressinPositive==None):
        for go_term in d["metadata"]["go_terms"]:
            if go_term["identifier"] in nonNegative_GOterms:
                bressinNegative = False
                # break
            if bressinPositive == False and go_term["identifier"] == "GO:0003723":
                print(f"WARNING: {accession} is has no molecule wide RNA bindign term but has a local one!")

    RBP_Name = d["metadata"]["gene"]  # e.g. PUM2
    # if(d["metadata"]["gene"] == "PUM2"):
    #    pprint(result)
    Protein_Sequences = d["metadata"]["sequence"]  # AA sequence
    Protein_ID = RBP_Name  # Protein ID (This will be the filename!)

    # get InterPro domains
    annotations = []  # list of triplets: (from, to, type, name)
    # type is either:
    # 0 = other-domains
    # 1 = RNA binding domain (Go annotation 'GO:0003723')
    # 2 = IDR
    # name is more specific
    # e.g. for type=1: PUM-HD
    # or for type=2: Mobidblt-Consensus Disorder

    for annotation in d["entry_subset"]:
        if annotation["entry_type"] != "domain":  # domains for RBD or other
            # HIER: somehow get the "other Features" sogement from https://www.ebi.ac.uk/interpro/protein/UniProt/Q8TB72/
            # and the MobiDB-lite IDR prediction"
            continue

        # print(f"annotation:{annotation}")
        ty, name, sName = IPR_getDomainInfos(annotation["accession"], APIroot)

        # get regions
        for region in annotation["entry_protein_locations"]:
            fragement = region["fragments"][0]  # take first fragment, what do the other fragemtns mean?! (TODO)
            fr = fragement["start"]
            to = fragement["end"]

            annotations.append((fr, to, ty, name, sName))

            if len(region["fragments"]) > 1:
                print(
                    f'\t\t{accession}\t annotation["entry_protein_locations"][0]["fragments"]: {annotation["entry_protein_locations"][0]["fragments"]}'
                )

    # if len(prot_RBDs) == 0:
    #    if not silent:
    #        print(f'\t\t{accession}\t No RBDs found')
    # pprint(f'\t\t{accession}\t No domains found: {d["entry_subset"]}')

    # get MobiDB-lite annotations
    for IDR in getMobiIDRs(accession):
        fr, to = IDR
        annotations.append((fr, to, 2, "IDR", "MobiDB-lite IDR"))

    Canonical = None  # TODO: get canonical information

    print(f"{accession}\t{bressinPositive} (Thread)")

    return accession, Species, RBP_Name, Protein_Sequences, Protein_ID, annotations, Canonical, bressinPositive, bressinNegative


# gets dataset from InterPro
def getInterPro(
    dataFolder,
    filePath,
    onlyKnownRBPs=False,
    fromWeb=False,
    forceRefresh=False,
    APIroot="https://www.ebi.ac.uk:443/interpro/api/",
    silent=True,
    threads=16,
):
    if forceRefresh or (os.path.isfile(filePath) == False):
        # get list of relevant proteins (might include duplicates)
        if onlyKnownRBPs:
            results = IP_with_RBD(APIroot=APIroot, silent=silent)

            # fileName=RBPfilePath#"InterPro_RBPs_human_reviewed_knownRBDs.pkl"
        else:
            # get all 21k reviewd (swiss prod) proteins for human
            # "https://www.ebi.ac.uk:443/interpro/api/protein/reviewed/taxonomy/uniprot/9606/?page_size=200"
            results = IP_all_RBP(
                dataFolder=dataFolder,
                fromWeb=fromWeb,  # Because the web APi returns too many fale positive proteins (GO filtering does not work!)
                APIroot=APIroot,
                silent=silent,
            )

            # fileName=RBPfilePath#"InterPro_RBPs_human_reviewed_allRBDs.pkl"

        # NOTE: at this point "results" can include accessions that actually have no corresponding InterPro entry!
        # This is why later we filter out based on wheter all other queries were sucessful

        # check/filter duplicats
        print(f"Filter duplicates")
        accession_seen = []
        results_filtered = []
        for result in results:
            accession = result["metadata"]["accession"]
            if accession in accession_seen:
                print(f"\tAcession allready seen: {accession}")
            else:
                results_filtered.append(result)
                accession_seen.append(accession)
        results = results_filtered
        print(f"Results (duplicates removed): {len(results)}")

        # prepare dataframe structure
        Gene_IDs = []  # Gene ID  / Accession (will be the filename in the embeedings folder)
        Species = []  # Species (i.e. 9606)
        RBP_Names = []  # RBP name
        Protein_IDs = []  # Protein ID
        Protein_Sequences = []  # AA sequence
        Domains = []  # triplet: (from, to, type)
        Canonical = []  # boolean (or None if not known)
        BressinPositive = []  # is the protein in the bnding positive set (according to bressin 19 critera)
        BressinNegative = []  # is the protein in the bnding negative set (according to bressin 19 critera)

        # iterate trough all proteins and get relevant data
        print(f"Get data for each protein")
        with tqdm(total=len(results)) as pbar:
            for batch_index in range(0, len(results), threads):  # work in batches
                batch = results[batch_index : batch_index + threads]

                # create input data
                threadData = []
                for result in batch:
                    # append data
                    threadData.append((result, APIroot))

                # execture paralell processes
                with Pool(len(threadData)) as p:
                    returnData = p.map(process_getInterPro, threadData)

                # extract output data
                for i, dataSet in enumerate(returnData):  # the return datat ordering might be different from the trhead data ordering
                    accession, species, RBP_Name, seq, Protein_ID, annotations, canonical, bressinPositive, bressinNegative = dataSet

                    # APPLY filters
                    if seq == None:  # we could not get all data (acession is probably not in database)
                        # print(f"{accession}: not found for deeper queries (domains, sequence, etc.)")
                        continue  # we do not need these proteins

                    if bressinPositive == bressinNegative:  # remove entries where pos and neg seperation is not clear yet
                        continue
                    # df = RBPdomains_IG_havingAttrib
                    # total = len(df)
                    # onlyBressinPos = sum(np.logical_and(df.bressinPositive==True, df.bressinNegative==False))
                    # onlyBressinNeg = sum(np.logical_and(df.bressinPositive==False, df.bressinNegative==True))
                    # bressinBoth = sum(np.logical_and(df.bressinPositive==True, df.bressinNegative==True))
                    # bressinNone = sum(np.logical_and(df.bressinPositive==False, df.bressinNegative==False))
                    # print(f"total: {total}")
                    # print(f"bressin +: {onlyBressinPos}\t({(onlyBressinPos/total)*100:.2f}%)")
                    # print(f"bressin -: {onlyBressinNeg}\t({(onlyBressinNeg/total)*100:.2f}%)")
                    # print(f"bressin +,-: {bressinBoth}\t({(bressinBoth/total)*100:.2f}%)")
                    # print(f"bressin ?: {bressinNone}\t({(bressinNone/total)*100:.2f}%)")
                    # RBPdomains_IG = RBPdomains_IG_havingAttrib.loc[df.bressinPositive!=df.bressinNegative]
                    # print(f"-> have only one class: {len(RBPdomains_IG)}\t({(len(RBPdomains_IG)/total)*100:.2f}%)")

                    Gene_IDs.append(accession)
                    Species.append(species)
                    RBP_Names.append(RBP_Name)
                    Protein_IDs.append(Protein_ID)
                    Protein_Sequences.append(seq)
                    Domains.append(annotations)
                    Canonical.append(canonical)
                    BressinPositive.append(bressinPositive)
                    BressinNegative.append(bressinNegative)

                pbar.update(len(batch))

        if not silent:
            print("Create Dataframe")

        RBPs = pd.DataFrame(
            data={
                "Gene_ID": Gene_IDs,
                "Species": Species,
                "RBP_Name": RBP_Names,
                "Protein_ID": Protein_IDs,
                "Protein_seq": Protein_Sequences,
                "domains": Domains,
                "canonical": Canonical,
                "positive": BressinPositive,
                "negative": BressinNegative,
            }
        )

        if not silent:
            print(f"InterPro: Saving RBPs to {filePath}")
        RBPs.to_pickle(filePath)

    else:
        if not silent:
            print(f"InterPro: Loading RBPs from {filePath}")
        RBPs = pd.read_pickle(filePath)

    return RBPs


##### UniProd #####


# get all RBPs from uniProd that are: canonical + reviewed + human + RNA-binding (GO) proteins -> 1667
def UP_all_RBP(
    dataFolder,
    fromWeb=True,  # if you want to use the webquery (otehrweise oyu need to have downloaded the json file!)
    species_id=9606,
    silent=True,
):
    # get RBP names
    if fromWeb:
        url = f"https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=json&query=%28%28go%3A0003723%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28taxonomy_id%3A{species_id}%29%29"

        if not silent:
            print(f"Get from {url}")

        ret = requests.get(url)
        if ret.status_code == 200:
            d = json.loads(ret.text)
        results = d["results"]

    else:  # offline file
        filepath = dataFolder + "uniProd_human_reviewed_GOrna-binding_canonical/file_1667.json"

        print(f"Loading file {filepath}")
        with open(filepath) as f:
            d = json.load(f)
        results = d["results"]

    return results


# DEPRECIATED! does not follow the new interface (InterPro uses)
def getUniProd(dataFolder, filePath, onlyKnownRBPs=False, forceRefresh=False, silent=True):
    if forceRefresh or (os.path.isfile(filePath) == False):
        # get list of relevant proteins (might include duplicates)
        if onlyKnownRBPs:
            raise NotImplementedError("We do not yet have a way to filter what domains are binding an which not")

            results = UP_all_RBP(dataFolder=dataFolder, fromWeb=True, silent=silent)
            results = UP_with_RBP()

            # fileName="UniProd_RBPs_human_reviewed_knownRBDs.pkl"
        else:
            results = UP_all_RBP(dataFolder=dataFolder, fromWeb=True, silent=silent)
            # fileName=RBPfilePath#"UniProd_RBPs_human_reviewed_allRBDs.pkl"

        # prepare dataframe structure
        Species = []  # Species (i.e. 9606)
        RBP_Names = []  # RBP name
        Gene_IDs = []  # Gene ID  / Accession
        Protein_IDs = []  # Protein ID (will be the filename in the embeedings folder)
        Protein_Sequences = []  # AA sequence
        Domains = []  # triplet: (from, to, type)
        Canonical = []  # boolean (or None if not known)

        # extract (binding) domains and disordered regions
        if not silent:
            print(f"Extracting binding domains and IDRs")

        for i, result in enumerate(tqdm(results)):
            RBP_Name = result["uniProtkbId"][:-6]  # remove the "_HUMAN" at the end because it all human we are currently looking at
            accession = result["primaryAccession"]

            if accession in Gene_IDs:  # if we allready have this protein in our database
                # print(f"\t\t{accession}\t -> allready in list")
                continue

            Gene_IDs.append(accession)
            RBP_Names.append(RBP_Name)
            Species.append(result["organism"]["taxonId"])
            Protein_Sequences.append(result["sequence"]["value"])

            Protein_IDs.append(RBP_Name)  # Protein ID (This will be the filename!) #TODO: haben wir da besser infos?

            # get Domains
            prot_RBDs = []
            prot_IDRs = []
            for entry in result["features"]:
                if entry["type"] == "Domain":
                    location = entry["location"]
                    fr = location["start"]["value"]
                    to = location["end"]["value"]
                    description = entry["description"]

                    triple = (fr, to, description)
                    prot_RBDs.append(triple)
                    # print(f'\t{i}: dom = {triple}')

                elif (
                    entry["type"] == "Region" and entry["description"] == "Disordered"
                ):  # all entry["description"] == "Disordered" are regions
                    location = entry["location"]
                    fr = location["start"]["value"]
                    to = location["end"]["value"]

                    triple = (fr, to, "IDR")
                    # triple = (fr,to,sources)
                    prot_IDRs.append(triple)

                    ## SOURCE:
                    # is normally ModiDB
                    # Proteins that have different soruce (PubMed) for IDRs are: 77, 279, 462, 921

                    # sources = [(evid["source"],evid["id"])for evid in entry["evidences"]]

                    # print(f'\t{i}: dis = {triple}')

                else:  # uniinteresting feature
                    # print(f"Other: {entry['description']}")

                    # what other MobiDB-lite annotations are there?
                    # Spoiler: only "Compositional bias" types
                    # if("evidences" in entry.keys()):
                    #    for evid in entry["evidences"]:
                    #        if "id" in evid.keys() and evid["id"] == "MobiDB-lite":
                    #            print(f'{i}: MobiDB-lite = type={entry["type"]}, description={entry["description"]}')

                    continue

            # TODO: better postprocessing of IDRs and RBDs possible? does it make sense to make on list out of them?
            domains = prot_RBDs + prot_IDRs  # add normal domains as well as IDRs togetehr to one list
            Domains.append(domains)

            # if(domains==[]):
            #    print(f'{i}: {accession} {RBP_Name}:\t{prot_RBDs}')
            #    pprint(result["features"])

            Canonical.append(True)  # UniProd entries are allways canonical

        if not silent:
            print("Create Dataframe")

        RBPs = pd.DataFrame(
            data={
                "Species": Species,
                "RBP_Name": RBP_Names,
                "Gene_ID": Gene_IDs,
                "Protein_ID": Protein_IDs,
                "Protein_seq": Protein_Sequences,
                "domains": Domains,
                "canonical": Canonical,
            }
        )

        if not silent:
            print(f"UniProd: Saving RBPs to {filePath}")
        RBPs.to_pickle(filePath)

    else:
        if not silent:
            print(f"UniProd: Loading RBPs from {filePath}")
        RBPs = pd.read_pickle(filePath)

    return RBPs


##### RIC ######

RICcolumns = {  # file name (without extension) : (speciesID, [relevantColumns])
    # ARATH - Arabidopsis thaliana
    "RBPbase_At_DescriptiveID": (3702, ["RBPBASE000000001.1", "RBPBASE000000002.1", "RBPBASE000000003.1"]),
    # 9HYME - Chrysis elegans #TODO: is that the correct entry?
    # "RBPbase_Ce_DescriptiveID": ( 212608, [ ])
    # 9MUSC - Drosophila #TODO: is that the correct entry?
    "RBPbase_Dm_DescriptiveID": (7215, ["RBPBASE000000005.1", "RBPBASE000000006.1"]),
    # HUMAN - Homo sapiens
    "RBPbase_Hs_DescriptiveID": (
        9606,
        [
            "RBPBASE000000008.1",
            "RBPBASE000000009.1",
            "RBPBASE000000034.1",
            "RBPBASE000000035.1",
            "RBPBASE000000059.1",
            "RBPBASE000000060.1",
            "RBPBASE000000061.1",
            "RBPBASE000000062.1",
        ],
    ),
    # MOUSE - Mus musculus
    "RBPbase_Mm_DescriptiveID": (
        10090,
        [
            "RBPBASE000000014.1",
            "RBPBASE000000016.1",
            "RBPBASE000000017.1",
            "RBPBASE000000019.1",
            "RBPBASE000000053.1",
            "RBPBASE000000054.1",
            "RBPBASE000000055.1",
        ],
    ),
    # YEAST - accharomyces cerevisiae (strain ATCC 204508 / S288c)
    "RBPbase_Sc_DescriptiveID": (559292, ["RBPBASE000000020.1" "RBPBASE000000024.1"]),
}


def getRIC(
    dataFolder,
    filePath,
    RICpath,
    thresholdPos=3,  # if three columns are positive
    onlyKnownRBPs=False,
    forceRefresh=False,
    silent=True,
    threads=16,
):
    global RICcolumns

    if forceRefresh or (os.path.isfile(filePath) == False):
        raw = pd.read_csv(str(RICpath), sep="\t", header=0, encoding="latin-1")

        if not RICpath.stem in RICcolumns.keys():
            raise RuntimeError(f'Unknwon file stem for RIC files "{RICpath.stem}" Known stems: [{list(RICcolumns.keys())}]')
        else:
            Species, relevantColumnNames = RICcolumns[RICpath.stem]

            relevantColumnNames_real = []
            for realName in raw.keys():
                for targetname in relevantColumnNames:
                    if targetname in realName:
                        relevantColumnNames_real.append(realName)

            relevantColumnNames = relevantColumnNames_real

        RBPs = {
            "Gene_ID": raw["UnitProtSwissProtID-Hs\nRBPANNO000000043.1"],  # uniprod Gene ID e.g. Q8TB72
            "RBP_Name": raw["UNIQUE"],  # e.g. PUM2
            # raw["ID"], e.g.  ENSG00000055917 for PUM2
            "Species": [Species] * len(raw),
        }

        # Count Positive Tests
        counterVector = np.zeros(len(raw), dtype=int)
        for key in relevantColumnNames:
            columnData = raw[key] == "YES"

            counterVector += columnData

        RBPs["posTestCount"] = counterVector

        # Report
        print(f"Relevant columns: {len(relevantColumnNames)} ({relevantColumnNames})\nThreshold: {thresholdPos}")
        total = len(raw)
        cum = 0
        print("pos\t#\t%\tcumu\tcumu%")
        for c, v in np.transpose(np.unique(counterVector, return_counts=True)):
            r = total - cum
            print(f"{c}\t{v} \t{(v/total)*100:.3f}%\t{r}\t{(r/total)*100:.3f}%")
            cum += v

        # Threshold
        RBPs["positive"] = counterVector >= thresholdPos

        # Report
        print(f"Thresholding with >= {thresholdPos} positives:")
        pos = sum(RBPs["positive"])
        print(f"positive: {pos}")
        print(f"negative: {total-pos}")

        # TODO:
        # - Protein_seq (get from???)
        # - domains
        # - canonical

        if not silent:
            print("Create Dataframe")

        RBPs = pd.DataFrame(RBPs)
        if not silent:
            print(f"InterPro: Saving RBPs to {filePath}")
        RBPs.to_pickle(filePath)

    else:
        if not silent:
            print(f"InterPro: Loading RBPs from {filePath}")
        RBPs = pd.read_pickle(filePath)

    return RBPs


##### Bressin #####
# the original set Peng et al. 19 used


def getBressin19(
    dataFolder,
    filePath,
    fileNameRBP,
    fileNameNRBP,
    forceRefresh=False,
    silent=True,
):
    if forceRefresh or (os.path.isfile(filePath) == False):
        RBPs = {"Gene_ID": [], "RBP_Name": [], "Species": [], "Protein_seq": [], "positive": []}

        for fileName, pos in [(fileNameRBP, True), (fileNameNRBP, False)]:
            print(fileName)
            with open(fileName, "r") as f:
                lines = f.readlines()

            for description, sequence in list(zip(lines[::2], lines[1::2])):
                sequence = sequence[:-1]
                description = description[:-1]
                split = description.split("|")

                RBPs["Gene_ID"].append(split[1])

                if split[2][-6:] == "_HUMAN":  # TODO: this only works for "_HUMAN"!
                    RBPs["RBP_Name"].append(split[2][:-6])
                else:
                    RBPs["RBP_Name"].append(split[2])

                RBPs["Species"].append(9606)
                RBPs["Protein_seq"].append(str(sequence))
                RBPs["positive"].append(pos)

        if not silent:
            print("Create Dataframe")

        RBPs = pd.DataFrame(RBPs)
        if not silent:
            print(f"Bressin19: Saving RBPs to {filePath}")
        RBPs.to_pickle(filePath)

    else:
        if not silent:
            print(f"Bressin19: Loading RBPs from {filePath}")
        RBPs = pd.read_pickle(filePath)

    return RBPs
