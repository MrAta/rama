use super::{MutView, Storage, View};
#[derive(Debug, Clone, Default)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub scale: f32,
    pub zero_point: i32,

}

#[derive(Default)]
pub struct RunState<T: Storage> {
    pub x: T,
    pub xb: T,
    pub xb2: T,
    pub hb: T,
    pub hb2: T,
    pub q: T,
    pub k: T,
    pub v: T,
    pub att: T,
    pub logits: T,
    pub key_cache: T,
    pub value_cache: T,
}

#[derive(Default)]
pub struct QRunState<T: Storage> {
    pub x: T,
    pub xq: QuantizedTensor,
    pub xb: T,
    pub xb2: T,
    pub hb: T,
    pub hb2: T,
    pub hq: QuantizedTensor,
    pub q: T,
    pub k: T,
    pub v: T,
    pub att: T,
    pub logits: T,
    pub key_cache: T,
    pub value_cache: T,
}

pub struct RunStateView<'a, T: Storage> {
    pub x: MutView<'a, T>,
    pub xb: MutView<'a, T>,
    pub xb2: MutView<'a, T>,
    pub hb: MutView<'a, T>,
    pub hb2: MutView<'a, T>,
    pub q: MutView<'a, T>,
    pub k: MutView<'a, T>,
    pub v: MutView<'a, T>,
    pub att: MutView<'a, T>,
    pub logits: MutView<'a, T>,
    pub key_cache: MutView<'a, T>,
    pub value_cache: MutView<'a, T>,
}

pub struct QRunStateView<'a, T: Storage> {
    pub x: MutView<'a, T>,
    pub xq: QuantizedTensor,
    pub xb: MutView<'a, T>,
    pub xb2: MutView<'a, T>,
    pub hb: MutView<'a, T>,
    pub hb2: MutView<'a, T>,
    pub hq: QuantizedTensor,
    pub q: MutView<'a, T>,
    pub k: MutView<'a, T>,
    pub v: MutView<'a, T>,
    pub att: MutView<'a, T>,
    pub logits: MutView<'a, T>,
    pub key_cache: MutView<'a, T>,
    pub value_cache: MutView<'a, T>,
}

impl<'a, T: Storage> RunStateView<'a, T> {
    pub fn from_rs(rs: &mut RunState<T>) -> RunStateView<'_, T> {
        RunStateView {
            x: MutView::new(&mut rs.x),
            xb: MutView::new(&mut rs.xb),
            xb2: MutView::new(&mut rs.xb2),
            hb: MutView::new(&mut rs.hb),
            hb2: MutView::new(&mut rs.hb2),
            q: MutView::new(&mut rs.q),
            k: MutView::new(&mut rs.k),
            v: MutView::new(&mut rs.v),
            att: MutView::new(&mut rs.att),
            logits: MutView::new(&mut rs.logits),
            key_cache: MutView::new(&mut rs.key_cache),
            value_cache: MutView::new(&mut rs.value_cache),
        }
    }
}

impl <'a, T: Storage> QRunStateView<'a, T> {
    pub fn from_qrs(qrs: &mut QRunState<T>) -> QRunStateView<'_, T> {
        QRunStateView {
            x: MutView::new(&mut qrs.x),
            xq: qrs.xq.clone(),
            xb: MutView::new(&mut qrs.xb),
            xb2: MutView::new(&mut qrs.xb2),
            hb: MutView::new(&mut qrs.hb),
            hb2: MutView::new(&mut qrs.hb2),
            hq: qrs.hq.clone(),
            q: MutView::new(&mut qrs.q),
            k: MutView::new(&mut qrs.k),
            v: MutView::new(&mut qrs.v),
            att: MutView::new(&mut qrs.att),
            logits: MutView::new(&mut qrs.logits),
            key_cache: MutView::new(&mut qrs.key_cache),
            value_cache: MutView::new(&mut qrs.value_cache),
        }
    }
}

// Transformer Weights
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TransformerWeights<T: Storage> {
    pub token_embedding_table: T,
    pub rms_att_weight: T,
    pub rms_ffn_weight: T,

    pub wq: T,
    pub wk: T,
    pub wv: T,
    pub wo: T,
    pub w1: T,
    pub w2: T,
    pub w3: T,

    pub rms_final_weight: T,
    pub freq_cis_real: T,
    pub freq_cis_imag: T,
    pub wcls_exists: bool,
    pub wcls: T,
}

// Quantized Transformer Weights
#[repr(C)]
#[derive(Debug, Clone)]
pub struct QTransformerWeights<T: Storage> {
    pub q_tokens: QuantizedTensor,
    pub token_embedding_table: T,
    pub rms_att_weight: T,
    pub rms_ffn_weight: T,

    pub wq: QuantizedTensor,
    pub wk: QuantizedTensor,
    pub wv: QuantizedTensor,
    pub wo: QuantizedTensor,
    pub w1: QuantizedTensor,
    pub w2: QuantizedTensor,
    pub w3: QuantizedTensor,

    pub rms_final_weight: T,
    pub freq_cis_real: T,
    pub freq_cis_imag: T,
    pub wcls_exists: bool,
    pub wcls: QuantizedTensor,
}

pub struct TransformerWeightsView<'a, T: Storage> {
    pub token_embedding_table: View<'a, T>,
    pub rms_att_weight:  View<'a, T>,
    pub rms_ffn_weight: View<'a, T>,
    pub wq: View<'a, T>,
    pub wk: View<'a, T>,
    pub wv: View<'a, T>,
    pub wo: View<'a, T>,
    pub w1: View<'a, T>,
    pub w2: View<'a, T>,
    pub w3: View<'a, T>,
    pub rms_final_weight:View<'a, T>,
    pub freq_cis_real: View<'a, T>,
    pub freq_cis_imag: View<'a, T>,
    pub wcls_exists: bool,
    pub wcls: View<'a, T>,
}

pub struct QTransformerWeightsView<'a, T: Storage> {
    pub q_tokens: QuantizedTensor,
    pub token_embedding_table: View<'a, T>,
    pub rms_att_weight:  View<'a, T>,
    pub rms_ffn_weight: View<'a, T>,
    pub wq: QuantizedTensor,
    pub wk: QuantizedTensor,
    pub wv: QuantizedTensor,
    pub wo: QuantizedTensor,
    pub w1: QuantizedTensor,
    pub w2: QuantizedTensor,
    pub w3: QuantizedTensor,
    pub rms_final_weight:View<'a, T>,
    pub freq_cis_real: View<'a, T>,
    pub freq_cis_imag: View<'a, T>,
    pub wcls_exists: bool,
    pub wcls: QuantizedTensor,
}

impl<'a> TransformerWeightsView<'a, Vec<f32>> {
    #[allow(dead_code)]
    pub fn from_ws(ws: &'a TransformerWeights<Vec<f32>>) -> TransformerWeightsView<'a, Vec<f32>> {
        TransformerWeightsView {
            token_embedding_table: View::new(&ws.token_embedding_table),
            rms_att_weight: View::new(&ws.rms_att_weight),
            rms_ffn_weight: View::new(&ws.rms_ffn_weight),
            wq: View::new(&ws.wq),
            wk: View::new(&ws.wk),
            wv: View::new(&ws.wv),
            wo: View::new(&ws.wo),
            w1: View::new(&ws.w1),
            w2: View::new(&ws.w2),
            w3: View::new(&ws.w3),
            rms_final_weight: View::new(&ws.rms_final_weight),
            freq_cis_real: View::new(&ws.freq_cis_real),
            freq_cis_imag: View::new(&ws.freq_cis_imag),
            wcls: {
                if ws.wcls_exists {
                    View::new(&ws.wcls)
                } else {
                    View::new(&ws.token_embedding_table)
                }
            },
            wcls_exists: ws.wcls_exists,
        }

    }
}

impl <'a > QTransformerWeightsView<'a, Vec<f32>> {
    #[allow(dead_code)]
    pub fn from_qws(qws: &'a QTransformerWeights<Vec<f32>>) -> QTransformerWeightsView<'a, Vec<f32>> {
        QTransformerWeightsView {
            q_tokens: qws.q_tokens.clone(),
            token_embedding_table: View::new(&qws.token_embedding_table),
            rms_att_weight: View::new(&qws.rms_att_weight),
            rms_ffn_weight: View::new(&qws.rms_ffn_weight),
            wq: qws.wq.clone(),
            wk: qws.wk.clone(),
            wv: qws.wv.clone(),
            wo: qws.wo.clone(),
            w1: qws.w1.clone(),
            w2: qws.w2.clone(),
            w3: qws.w3.clone(),
            rms_final_weight: View::new(&qws.rms_final_weight),
            freq_cis_real: View::new(&qws.freq_cis_real),
            freq_cis_imag: View::new(&qws.freq_cis_imag),
            wcls: qws.wcls.clone(),
            wcls_exists: qws.wcls_exists,
        }
    }
}