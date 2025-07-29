import pandas as pd
import numpy as np
from Bio import Entrez, SeqIO
import requests
from typing import List, Dict, Optional

class SickleCellDataAcquisition:
    def __init__(self, email: str):
        """
        Initialize the data acquisition class
        :param email: Your email address (required for NCBI API)
        """
        Entrez.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
    def get_hbb_gene_data(self) -> Dict:
        """
        Get HBB gene data from NCBI
        HBB is the gene that causes sickle cell anemia when mutated
        """
        # Search for HBB gene
        handle = Entrez.esearch(db="gene", term="HBB[Gene Name] AND human[Organism]")
        record = Entrez.read(handle)
        handle.close()
        
        if not record["IdList"]:
            raise ValueError("HBB gene not found")
            
        gene_id = record["IdList"][0]
        
        # Get gene information
        handle = Entrez.efetch(db="gene", id=gene_id, rettype="gb", retmode="text")
        gene_info = handle.read()
        handle.close()
        
        return {
            "gene_id": gene_id,
            "gene_info": gene_info
        }
        
    def get_sickle_cell_variants(self) -> pd.DataFrame:
        """
        Get known sickle cell variants from ClinVar
        """
        # Search ClinVar for sickle cell variants
        handle = Entrez.esearch(
            db="clinvar",
            term="sickle cell anemia[Condition] AND HBB[Gene]",
            retmax=1000
        )
        record = Entrez.read(handle)
        handle.close()
        
        variants = []
        for var_id in record["IdList"]:
            handle = Entrez.efetch(db="clinvar", id=var_id, rettype="vcv")
            var_record = Entrez.read(handle)
            handle.close()
            
            if "reference_clinical_assertion" in var_record:
                variant = var_record["reference_clinical_assertion"]
                variants.append({
                    "variant_id": var_id,
                    "clinical_significance": variant.get("clinical_significance", {}).get("description", ""),
                    "gene": variant.get("gene", {}).get("symbol", ""),
                    "mutation": variant.get("measure_set", {}).get("measure", [{}])[0].get("name", ""),
                    "chromosome": variant.get("measure_set", {}).get("measure", [{}])[0].get("chromosome", ""),
                    "position": variant.get("measure_set", {}).get("measure", [{}])[0].get("position", "")
                })
                
        return pd.DataFrame(variants)
        
    def get_expression_data(self) -> pd.DataFrame:
        """
        Get HBB gene expression data from GEO
        """
        # Search GEO for HBB expression studies
        handle = Entrez.esearch(
            db="gds",
            term="HBB[Gene] AND expression[Filter] AND human[Organism]",
            retmax=100
        )
        record = Entrez.read(handle)
        handle.close()
        
        expression_data = []
        for gds_id in record["IdList"]:
            handle = Entrez.efetch(db="gds", id=gds_id)
            gds_record = Entrez.read(handle)
            handle.close()
            
            # Extract relevant information
            expression_data.append({
                "study_id": gds_id,
                "title": gds_record["title"],
                "summary": gds_record["summary"],
                "platform": gds_record["platform"],
                "samples": gds_record["samples"]
            })
            
        return pd.DataFrame(expression_data)
        
    def get_sequence_data(self) -> Dict:
        """
        Get HBB gene sequence data
        """
        # Get HBB gene sequence
        handle = Entrez.efetch(
            db="nucleotide",
            id="NC_000011.10",
            rettype="fasta",
            seq_start=5225464,
            seq_stop=5227071
        )
        sequence = handle.read()
        handle.close()
        
        return {
            "sequence": sequence,
            "chromosome": "11",
            "start": 5225464,
            "end": 5227071
        }
        
    def get_patient_data(self) -> pd.DataFrame:
        """
        Get sickle cell patient data from TCGA (if available)
        """
        # Note: This is a placeholder. Actual TCGA data access requires proper authentication
        # and compliance with data usage agreements
        return pd.DataFrame({
            "patient_id": [],
            "mutation_type": [],
            "clinical_outcome": [],
            "treatment_response": []
        })
        
    def get_all_data(self) -> Dict:
        """
        Get all relevant data for sickle cell anemia analysis
        """
        return {
            "gene_data": self.get_hbb_gene_data(),
            "variants": self.get_sickle_cell_variants(),
            "expression": self.get_expression_data(),
            "sequence": self.get_sequence_data(),
            "patient_data": self.get_patient_data()
        } 