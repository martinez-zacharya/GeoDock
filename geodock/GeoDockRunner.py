import esm
import torch
from time import time
from geodock.utils.embed import embed
from geodock.utils.docking import dock
from geodock.model.interface import GeoDockInput
from geodock.model.GeoDock import GeoDock
from esm.inverse_folding.util import load_coords
from trill.utils.lightning_models import ESM
from geodock.utils.embed import get_pair_mats, get_pair_relpos


class GeoDockRunner():
    """
    Wrapper for GeoDock model predictions.
    """
    def __init__(self, args, ckpt_file, verbose:bool=False):

        # Check if gpu is available
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')

        self.verbose = verbose

        # Load ESM-2 model
        self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_model.eval().to(self.device)  # disables dropout for deterministic results

        # Load GeoDock model
        self.model = GeoDock.load_from_checkpoint(ckpt_file, map_location=self.device).eval().to(self.device)

    def embed(
        self, 
        seq1, 
        seq2,
        coords1,
        coords2,
    ):
        start_time = time()
        embeddings = embed(
            seq1, 
            seq2,
            coords1,
            coords2,
            self.esm_model,
            self.batch_converter,
            self.device,
        )

        if self.verbose:
            print(f"Completed embedding in {time() - start_time:.2f} seconds.")

        return embeddings
    
    def dock(
        self, 
        partner1, 
        partner2, 
        out_name,
        do_refine=True,
        use_openmm=True,
    ):
        # Get seqs and coords
        coords1, seq1 = load_coords(partner1, chain=None)
        coords2, seq2 = load_coords(partner2, chain=None)
        coords1 = torch.nan_to_num(torch.from_numpy(coords1))
        coords2 = torch.nan_to_num(torch.from_numpy(coords2))

        # Get embeddings
        model_in = self.embed(
            seq1,
            seq2,
            coords1,
            coords2,
        )

        # Start docking
        dock(
            out_name,
            seq1,
            seq2,
            model_in,
            self.model,
            do_refine=do_refine,
            use_openmm=use_openmm,
        )

class EnMasseGeoDockRunner():
    """
    Wrapper for GeoDock model predictions.
    """
    def __init__(self, args, ckpt_file, verbose:bool=False):

        # Check if gpu is available
        self.device = torch.device("cuda" if int(args.GPUs) > 0 else "cpu")
        # self.device = torch.device('cpu')

        self.verbose = verbose

        # # Load ESM-2 model
        _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # self.batch_converter = alphabet.get_batch_converter()
        # model_import_name = f'esm.pretrained.esm2_t33_650M_UR50D()'
        

        # self.esm_model, alphabet = ESM()
        self.batch_converter = alphabet.get_batch_converter()
        # self.esm_model.eval().to(self.device)  # disables dropout for deterministic results

        # Load GeoDock model
        self.model = GeoDock.load_from_checkpoint(ckpt_file, map_location=self.device).eval().to(self.device)

    def embed(
        self, 
        seq1, 
        seq2,
        coords1,
        coords2,
    ):
        start_time = time()
        embeddings = embed(
            seq1, 
            seq2,
            coords1,
            coords2,
            self.esm_model,
            self.batch_converter,
            self.device,
        )

        if self.verbose:
            print(f"Completed embedding in {time() - start_time:.2f} seconds.")

        return embeddings
    
    def dock(
        self, 
        rec_info, 
        lig_info, 
        out_name,
        do_refine=True,
        use_openmm=True,
    ):
        # # Get seqs and coords
        # coords1, seq1 = load_coords(partner1, chain=None)
        # coords2, seq2 = load_coords(partner2, chain=None)
        # coords1 = torch.nan_to_num(torch.from_numpy(coords1))
        # coords2 = torch.nan_to_num(torch.from_numpy(coords2))

        # # Get embeddings
        # model_in = self.embed(
        #     seq1,
        #     seq2,
        #     coords1,
        #     coords2,
        # )
        rec_name, rec_seq, rec_coord, rec_emb = rec_info
        lig_name, lig_seq, lig_coord, lig_emb = lig_info
        # print(type(rec_coord))
        # print(rec_coord.shape)
        rec_coord = torch.Tensor(rec_coord)
        lig_coord = torch.Tensor(lig_coord)
        rec_emb = torch.Tensor(rec_emb).to(self.device)
        lig_emb = torch.Tensor(lig_emb).to(self.device)
        coords = torch.cat([rec_coord, lig_coord], dim=0)
        input_pairs = get_pair_mats(coords, len(rec_seq))
        input_contact = torch.zeros(*input_pairs.shape[:-1])[..., None] 
        pair_embeddings = torch.cat([input_pairs, input_contact], dim=-1).to(self.device)
        
        # Get positional embeddings
        positional_embeddings = get_pair_relpos(len(rec_seq), len(lig_seq)).to(self.device)

        embeddings = GeoDockInput(
        protein1_embeddings=rec_emb.unsqueeze(0),
        protein2_embeddings=lig_emb.unsqueeze(0),
        pair_embeddings=pair_embeddings.unsqueeze(0),
        positional_embeddings=positional_embeddings.unsqueeze(0),
    )
        # print(f'{rec_emb.unsqueeze(0).shape=}')
        # print(f'{rec_emb.unsqueeze(0)=}')
        # print(f'{lig_emb.unsqueeze(0).shape=}')
        # print(f'{pair_embeddings.unsqueeze(0)=}')
        # print(f'{pair_embeddings.unsqueeze(0).shape=}')
        # print(f'{positional_embeddings.unsqueeze(0)=}')
        # print(f'{positional_embeddings.unsqueeze(0).shape=}')
        # Start docking
        dock(
            out_name,
            rec_seq,
            lig_seq,
            embeddings,
            self.model,
            do_refine=do_refine,
            use_openmm=use_openmm,
        )

if __name__ == '__main__':
    ckpt_file = "weights/dips_0.3.ckpt"
    partner1 = "./data/test/a9_1a95.pdb1_3.dill_r_b_COMPLEX.pdb"
    partner2 = "./data/test/a9_1a95.pdb1_3.dill_l_b_COMPLEX.pdb"
    out_name = "test"

    geodock = GeoDockRunner(ckpt_file=ckpt_file)
    pred = geodock.dock(
        partner1=partner1, 
        partner2=partner2,
        out_name=out_name,
        do_refine=True,
        use_openmm=True,
    )
