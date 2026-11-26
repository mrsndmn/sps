"""Baseline construction, frozen source features and immutable input identities."""
from dataclasses import fields
import torch
from torch import nn
from src.experiments import mtp, spec_baselines
from scripts import experiment_suite as e
from scripts.mtp_common import load_reference, file_sha256, shifted_targets, validate_evaluation_inputs
from modeling.models.sps.mtp import SPSMTP
from modeling.spec_baselines import MedusaHeads, TinyAR, TinyARConfig


class MedusaTraining(nn.Module):
    def __init__(self,reference):
        super().__init__()
        self.encoder=SPSMTP(reference,1,backend='auto').requires_grad_(False).eval()
        self.config=reference.config
        self.heads=MedusaHeads(self.config.hidden_size,spec_baselines.BASELINE_MEDUSA_HEADS)

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def forward(self,tokens):
        with torch.no_grad():
            cache,docs=self.encoder._encode_real(tokens,None)
            positions=torch.arange(tokens.shape[1],device=tokens.device)[None].expand(tokens.shape[0],-1)
            features=self.encoder._predict(cache,docs,positions,return_hidden=True)[:,:,0]
        return self.heads(features,self.encoder.reference.lm_head)


def identities():
    return dict(source_sha256=file_sha256(mtp.MTP_SOURCE),
        data_manifest_sha256=file_sha256(mtp.MTP_DATA/'manifest.json'))


def construct(exp,device):
    if exp.recipe=='medusa':return MedusaTraining(load_reference(device)).to(device)
    if exp.recipe=='ar_tiny':
        source=torch.load(mtp.MTP_SOURCE,map_location='cpu',weights_only=False,mmap=True)
        args=dict(source['model_args'],**spec_baselines.BASELINE_AR_ARGS)
        args['block_size']=spec_baselines.BASELINE_CONTEXT
        config = TinyARConfig(**{field.name: args[field.name] for field in fields(TinyARConfig)})
        return TinyAR(config).to(device)
    raise ValueError('Not a trainable baseline')


def trainable_state(model,recipe):
    return model.heads.state_dict() if recipe=='medusa' else model.state_dict()


def restore(model,state,exp):
    if state.get('recipe')!=exp.recipe:raise RuntimeError('Baseline recipe mismatch')
    if exp.recipe=='medusa':model.heads.load_state_dict(state['model'])
    else:model.load_state_dict(state['model'])


def loss_for(logits,tokens,recipe,config):
    horizons=spec_baselines.BASELINE_MEDUSA_HEADS+1 if recipe=='medusa' else 1
    labels=shifted_targets(tokens,spec_baselines.BASELINE_CONTEXT,horizons,
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
                or state.get('source')!=str(mtp.MTP_SOURCE)):
            raise RuntimeError('Changed existing MTP control')
        return trained,None
    if trained.get('design_signature')!=e.design_signature():raise RuntimeError('Stale baseline checkpoint')
    state=torch.load(trained['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if (state.get('design_signature')!=trained['design_signature'] or state.get('recipe')!=exp.recipe
            or state.get('tokens')!=spec_baselines.BASELINE_STEPS*spec_baselines.BASELINE_STEP_TOKENS
            or state.get('step')!=spec_baselines.BASELINE_STEPS):
        raise RuntimeError('Incomplete or changed baseline checkpoint')
    for key in ('source_sha256','data_manifest_sha256'):
        if state.get(key)!=trained.get(key):raise RuntimeError('Baseline provenance mismatch')
    return trained,state


def validate_resume(state,signature,identity):
    step=state.get('step')
    if (state.get('design_signature')!=signature or any(state.get(k)!=v for k,v in identity.items())
            or not isinstance(step,int) or isinstance(step,bool) or not 0<=step<=spec_baselines.BASELINE_STEPS
            or state.get('tokens')!=step*spec_baselines.BASELINE_STEP_TOKENS):
        raise RuntimeError('Changed or inconsistent baseline resume identity')
    return step


def finalize_checkpoint(dest,exp,signature,identity):
    """Recover the final artifact even if a completed run stopped before publishing it."""
    import shutil
    latest=dest/'latest.pt'
    state=torch.load(latest,map_location='cpu',weights_only=False,mmap=True)
    if validate_resume(state,signature,identity)!=spec_baselines.BASELINE_STEPS or state.get('recipe')!=exp.recipe:
        raise RuntimeError('Cannot finalize incomplete or incompatible baseline checkpoint')
    final=dest/'draft_final.pt';temporary=final.with_suffix('.tmp')
    shutil.copyfile(latest,temporary)
    temporary.replace(final)
    return final
