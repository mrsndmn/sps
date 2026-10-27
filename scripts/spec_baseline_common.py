"""Baseline construction, frozen source features and immutable input identities."""
from types import SimpleNamespace
import torch
from torch import nn
import experiments as e
from scripts.mtp_common import load_reference, file_sha256, shifted_targets, validate_evaluation_inputs
from modeling.models.sps.mtp import SPSMTP
from modeling.spec_baselines import MedusaHeads, TinyAR


class MedusaTraining(nn.Module):
    def __init__(self,reference):
        super().__init__()
        self.encoder=SPSMTP(reference,1,backend='auto').requires_grad_(False).eval()
        self.config=reference.config
        self.heads=MedusaHeads(self.config.hidden_size,e.BASELINE_MEDUSA_HEADS)

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self,tokens):
        with torch.no_grad():
            cache,docs=self.encoder._encode_real(tokens,None)
            positions=torch.arange(tokens.shape[1],device=tokens.device)[None].expand(tokens.shape[0],-1)
            features=self.encoder._predict(cache,docs,positions,return_hidden=True)[:,:,0]
        return self.heads(features,self.encoder.reference.lm_head)


def identities():
    return dict(source_sha256=file_sha256(e.MTP_SOURCE),
        data_manifest_sha256=file_sha256(e.MTP_DATA/'manifest.json'))


def construct(exp,device):
    if exp.recipe=='medusa':return MedusaTraining(load_reference(device)).to(device)
    if exp.recipe=='ar_tiny':
        source=torch.load(e.MTP_SOURCE,map_location='cpu',weights_only=False,mmap=True)
        args=dict(source['model_args'],**e.BASELINE_AR_ARGS)
        args['block_size']=e.BASELINE_CONTEXT
        return TinyAR(SimpleNamespace(**args)).to(device)
    raise ValueError('Not a trainable baseline')


def trainable_state(model,recipe):
    return model.heads.state_dict() if recipe=='medusa' else model.state_dict()


def restore(model,state,exp):
    if state.get('recipe')!=exp.recipe:raise RuntimeError('Baseline recipe mismatch')
    if exp.recipe=='medusa':model.heads.load_state_dict(state['model'])
    else:model.load_state_dict(state['model'])


def loss_for(logits,tokens,recipe,config):
    horizons=e.BASELINE_MEDUSA_HEADS+1 if recipe=='medusa' else 1
    labels=shifted_targets(tokens,e.BASELINE_CONTEXT,horizons,
        eos=config.eos_token_id,pad=config.pad_token_id)
    if recipe=='medusa':labels=labels[:,:,1:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),
        labels.reshape(-1),ignore_index=-100,reduction='sum')/(labels!=-100).sum().clamp_min(1)


def checked_state(exp):
    marker=e.training_marker(exp)
    import json
    trained=json.loads(marker.read_text())
    validate_evaluation_inputs(trained)
    if exp.recipe=='existing_mtp':
        state=torch.load(trained['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        if (state.get('design_signature')!=trained['design_signature'] or state.get('horizons')!=exp.horizons
                or state.get('source')!=str(e.MTP_SOURCE)):
            raise RuntimeError('Changed existing MTP control')
        return trained,None
    if trained.get('design_signature')!=e.design_signature():raise RuntimeError('Stale baseline checkpoint')
    state=torch.load(trained['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if (state.get('design_signature')!=trained['design_signature'] or state.get('recipe')!=exp.recipe
            or state.get('tokens')!=e.BASELINE_STEPS*e.BASELINE_STEP_TOKENS
            or state.get('step')!=e.BASELINE_STEPS):
        raise RuntimeError('Incomplete or changed baseline checkpoint')
    for key in ('source_sha256','data_manifest_sha256'):
        if state.get(key)!=trained.get(key):raise RuntimeError('Baseline provenance mismatch')
    return trained,state


def validate_resume(state,signature,identity):
    step=state.get('step')
    if (state.get('design_signature')!=signature or any(state.get(k)!=v for k,v in identity.items())
            or not isinstance(step,int) or isinstance(step,bool) or not 0<=step<=e.BASELINE_STEPS
            or state.get('tokens')!=step*e.BASELINE_STEP_TOKENS):
        raise RuntimeError('Changed or inconsistent baseline resume identity')
    return step
