import time
import gradio as gr
from main import ai_model, vector_db, rag_chain, device, classify_image, get_advice

css = """
@import url('https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap');

:root {
  --dark-slate-grey: #335c67;
  --vanilla-custard: #fff3b0;
  --honey-bronze:    #e09f3e;
  --brown-red:       #9e2a2b;
  --night-bordeaux:  #540b0e;
  --white:           #ffffff;
  --shadow:          rgba(83, 11, 14, 0.12);
}

/* ── Global font ── */
*, *::before, *::after {
    font-family: 'Comic Sans MS', 'Comic Sans', 'Patrick Hand', cursive !important;
    box-sizing: border-box;
}

/* ── Background ── */
body, .gradio-container {
    background: linear-gradient(145deg, #fff9c4 0%, var(--vanilla-custard) 60%, #fef5d0 100%) !important;
    min-height: 100vh;
}

/* ── Strip Gradio default panel chrome entirely ── */
.gradio-container .gr-form,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.block.padded,
.block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Header ── */
.derm-header {
    text-align: center;
    padding: 32px 20px 18px;
}
.derm-header h1 {
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    color: var(--night-bordeaux) !important;
    margin: 0 0 6px;
    text-shadow: 3px 3px 0 var(--honey-bronze);
}
.derm-header p {
    color: var(--dark-slate-grey) !important;
    font-size: 1rem !important;
    margin: 0;
    opacity: 0.85;
}
.pulse-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: var(--honey-bronze);
    border-radius: 50%;
    margin-left: 8px;
    vertical-align: middle;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { transform: scale(1); opacity:1; }
    50%      { transform: scale(1.5); opacity:0.5; }
}

/* ── Column panels via elem_id ──
   Gradio renders elem_id as the id on the wrapping .svelte-* div's child,
   so we target #upload-col and #results-col which Gradio puts on the column wrapper ── */
#upload-col > .wrap,
#upload-col > div,
#results-col > .wrap,
#results-col > div {
    background: var(--white) !important;
    border-radius: 20px !important;
    border: 2.5px solid var(--honey-bronze) !important;
    box-shadow: 5px 5px 0 var(--honey-bronze), 0 14px 36px var(--shadow) !important;
    padding: 22px !important;
}

/* Fallback: direct children of the column id */
#upload-col, #results-col {
    background: var(--white) !important;
    border-radius: 20px !important;
    border: 2.5px solid var(--honey-bronze) !important;
    box-shadow: 5px 5px 0 var(--honey-bronze), 0 14px 36px var(--shadow) !important;
    padding: 22px !important;
}

/* ── Panel labels ── */
.panel-label {
    display: block;
    font-size: 0.68rem !important;
    font-weight: 900 !important;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--honey-bronze) !important;
    margin-bottom: 10px;
}

/* ── Image upload widget ── */
#upload-col .image-container,
#upload-col [data-testid="image"],
#upload-col .svelte-1l6rhwr {
    border: 2.5px dashed var(--honey-bronze) !important;
    border-radius: 14px !important;
    background: #fffdf0 !important;
    min-height: 220px !important;
}

/* ── Analyze button ── */
#analyze-btn {
    width: 100% !important;
    margin-top: 14px !important;
    padding: 13px 0 !important;
    background: var(--dark-slate-grey) !important;
    color: var(--vanilla-custard) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    border: 3px solid var(--dark-slate-grey) !important;
    border-radius: 12px !important;
    cursor: pointer;
    transition: all 0.18s ease;
    box-shadow: 4px 4px 0 var(--night-bordeaux);
}
#analyze-btn:hover {
    background: var(--night-bordeaux) !important;
    border-color: var(--night-bordeaux) !important;
    transform: translate(-2px, -2px);
    box-shadow: 6px 6px 0 var(--honey-bronze);
}
#analyze-btn:active {
    transform: translate(2px, 2px);
    box-shadow: 2px 2px 0 var(--night-bordeaux);
}

/* ── Progress bar ── */
#progress-wrap {
    width: 100%;
    height: 5px;
    background: #f5edb0;
    border-radius: 3px;
    margin-top: 10px;
    overflow: hidden;
    opacity: 0;
    transition: opacity 0.3s;
}
#progress-wrap.loading { opacity: 1; }
#progress-inner {
    height: 100%;
    width: 40%;
    background: linear-gradient(90deg, var(--dark-slate-grey), var(--honey-bronze), var(--dark-slate-grey));
    background-size: 200% 100%;
    border-radius: 3px;
    display: none;
    animation: shimmer 1.6s linear infinite;
}
#progress-wrap.loading #progress-inner { display: block; }
@keyframes shimmer {
    0%   { background-position: 200% center; }
    100% { background-position: -200% center; }
}

/* ── Full-page gear overlay ── */
#gear-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(255, 249, 196, 0.9);
    backdrop-filter: blur(4px);
    z-index: 9999;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 22px;
}
#gear-overlay.active { display: flex; }

.gear-stage {
    position: relative;
    width: 130px;
    height: 130px;
}
.gear-item {
    position: absolute;
    animation: spin linear infinite;
}
.gear-item svg { display: block; }
.gear-big   { width: 85px; height: 85px; top: 0;  left: 0;   animation-duration: 3.2s; }
.gear-small { width: 52px; height: 52px; top: 5px; left: 64px; animation-duration: 2s; animation-direction: reverse; }
@keyframes spin { to { transform: rotate(360deg); } }

.loading-label {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: var(--night-bordeaux) !important;
    letter-spacing: 1px;
}
.loading-dots::after {
    content: '';
    animation: dots 1.4s steps(4, end) infinite;
}
@keyframes dots {
    0%  { content: '';    }
    25% { content: '.';   }
    50% { content: '..';  }
    75% { content: '...'; }
}

/* ── Circular confidence ── */
.conf-outer {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 16px 0 8px;
}
.circ-bg  { fill:none; stroke:#f0e4a0; stroke-width:3.8; }
.circ-arc {
    fill: none;
    stroke-width: 3.2;
    stroke-linecap: round;
    animation: arc-in 1.1s cubic-bezier(.4,0,.2,1) forwards;
}
@keyframes arc-in { from { stroke-dasharray: 0 100; } }
.arc-high { stroke: #27865a; }
.arc-med  { stroke: var(--honey-bronze); }
.arc-low  { stroke: var(--brown-red); }
.circ-pct {
    fill: var(--night-bordeaux);
    font-family: 'Comic Sans MS', cursive;
    font-size: 0.54em;
    text-anchor: middle;
    font-weight: 900;
}
.circ-lbl {
    fill: var(--dark-slate-grey);
    font-family: 'Comic Sans MS', cursive;
    font-size: 0.26em;
    text-anchor: middle;
}
.disease-pill {
    margin-top: 12px;
    background: var(--night-bordeaux);
    color: var(--vanilla-custard) !important;
    padding: 6px 22px;
    border-radius: 30px;
    font-size: 1rem !important;
    font-weight: 900 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    box-shadow: 3px 3px 0 var(--honey-bronze);
    display: inline-block;
}
.malignant-pill {
    margin-top: 7px;
    background: var(--brown-red);
    color: #fff !important;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    display: inline-block;
    animation: pulse 1.5s ease-in-out infinite;
}

/* ── Advice cards ── */
.advice-stack { display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
.advice-card {
    background: linear-gradient(135deg, #fffef5 0%, #fffbe8 100%);
    border-radius: 14px;
    border-left: 5px solid var(--dark-slate-grey);
    padding: 14px 16px;
    box-shadow: 3px 3px 0 rgba(83,11,14,0.07);
    transition: transform 0.15s, box-shadow 0.15s;
}
.advice-card:hover { transform: translateX(4px); box-shadow: 5px 5px 0 rgba(83,11,14,0.11); }
.card-steps { border-left-color: var(--honey-bronze); }
.card-tips  { border-left-color: var(--brown-red); }
.card-head {
    font-size: 0.7rem !important;
    font-weight: 900 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--night-bordeaux) !important;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.card-body {
    font-size: 0.9rem !important;
    color: var(--dark-slate-grey) !important;
    line-height: 1.65 !important;
}

/* ── Placeholder ── */
.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 340px;
    gap: 14px;
    opacity: 0.5;
    text-align: center;
}
.ph-icon { font-size: 3.5rem; }
.ph-txt  {
    font-size: 0.95rem !important;
    color: var(--dark-slate-grey) !important;
    max-width: 200px;
    line-height: 1.5;
}

/* ── Time footer ── */
.time-footer {
    text-align: center;
    font-size: 0.76rem !important;
    color: var(--brown-red) !important;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px dashed var(--honey-bronze);
    opacity: 0.7;
}

/* ── Page footer ── */
.page-footer {
    text-align: center;
    padding: 14px 0 20px;
    font-size: 0.72rem !important;
    color: var(--night-bordeaux) !important;
    opacity: 0.55;
}

footer { display: none !important; }
"""

def _gear(color):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="{color}">'
        '<circle cx="50" cy="50" r="11"/>'
        '<path d="M42 12h16l3 10a30 30 0 0 1 7 4l10-3 8 14-8 8a30 30 0 0 1 0 8l8 8-8 14-10-3'
        'a30 30 0 0 1-7 4l-3 10H42l-3-10a30 30 0 0 1-7-4l-10 3L14 66l8-8a30 30 0 0 1 0-8l-8-8'
        ' 8-14 10 3a30 30 0 0 1 7-4z"/>'
        '</svg>'
    )

LOADER_JS = """
<script>
(function () {
    var ready = false;

    function showLoader() {
        var o = document.getElementById('gear-overlay');
        var p = document.getElementById('progress-wrap');
        if (o) o.classList.add('active');
        if (p) p.classList.add('loading');
    }
    function hideLoader() {
        var o = document.getElementById('gear-overlay');
        var p = document.getElementById('progress-wrap');
        if (o) o.classList.remove('active');
        if (p) p.classList.remove('loading');
    }

    function setup() {
        if (ready) return;

        // Gradio renders the button with elem_id on a wrapper; find the actual <button>
        var btnWrap = document.getElementById('analyze-btn');
        var btn = btnWrap ? (btnWrap.tagName === 'BUTTON' ? btnWrap : btnWrap.querySelector('button')) : null;
        var resultsCol = document.getElementById('results-col');

        if (!btn || !resultsCol) {
            setTimeout(setup, 400);
            return;
        }
        ready = true;

        btn.addEventListener('click', showLoader);

        var observer = new MutationObserver(function(mutations) {
            for (var i = 0; i < mutations.length; i++) {
                if (mutations[i].addedNodes.length || mutations[i].type === 'characterData') {
                    hideLoader();
                    break;
                }
            }
        });
        observer.observe(resultsCol, { childList: true, subtree: true, characterData: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { setTimeout(setup, 600); });
    } else {
        setTimeout(setup, 600);
    }
})();
</script>
"""

GEAR_HTML = f"""
<div id="gear-overlay">
  <div class="gear-stage">
    <div class="gear-item gear-big">{_gear('#335c67')}</div>
    <div class="gear-item gear-small">{_gear('#e09f3e')}</div>
  </div>
  <div class="loading-label">Analyzing<span class="loading-dots"></span></div>
</div>
"""

MALIGNANT_DISEASES = {"melanoma", "basal cell carcinoma", "basal cell ca."}

def _ph(icon, text):
    return f"<div class='placeholder'><div class='ph-icon'>{icon}</div><div class='ph-txt'>{text}</div></div>"

PH_CIRC = _ph("🩺", "Confidence score will appear here after analysis")
PH_RES  = _ph("📄", "Upload an image and click Analyze to see AI recommendations")

def analyze_with_ui(image):
    if image is None:
        return PH_CIRC, PH_RES

    t0 = time.time()

    prediction = classify_image(ai_model, image, device)
    disease    = prediction["disease"]
    conf       = prediction["confidence"]
    conf_pct   = int(conf * 100)
    is_mal     = (
        disease.lower() in MALIGNANT_DISEASES
        or prediction.get("is_malignant", False)
    )

    result = get_advice(
        disease_name=disease,
        confidence=conf,
        vector_db=vector_db,
        rag_chain=rag_chain,
    )
    elapsed = time.time() - t0

    # Confidence arc
    if conf_pct >= 80:
        arc_cls, conf_word = "arc-high", "High Confidence"
    elif conf_pct >= 50:
        arc_cls, conf_word = "arc-med",  "Moderate"
    else:
        arc_cls, conf_word = "arc-low",  "Low Confidence"

    mal_badge = (
        "<div class='malignant-pill'>⚠ Potentially Malignant</div>" if is_mal else ""
    )

    circle_html = f"""
    <div class='conf-outer'>
      <svg viewBox='0 0 36 36' style='width:160px;height:160px;overflow:visible;'>
        <path class='circ-bg'
          d='M18 2.08 a 15.92 15.92 0 0 1 0 31.84 a 15.92 15.92 0 0 1 0 -31.84'/>
        <path class='circ-arc {arc_cls}'
          stroke-dasharray='{conf_pct},100'
          d='M18 2.08 a 15.92 15.92 0 0 1 0 31.84 a 15.92 15.92 0 0 1 0 -31.84'/>
        <text x='18' y='17.5' class='circ-pct'>{conf_pct}%</text>
        <text x='18' y='22.5' class='circ-lbl'>{conf_word}</text>
      </svg>
      <div class='disease-pill'>{disease}</div>
      {mal_badge}
    </div>"""

    def card(icon, title, body, extra=""):
        return (
            f"<div class='advice-card {extra}'>"
            f"<div class='card-head'><span>{icon}</span>{title}</div>"
            f"<div class='card-body'>{body}</div></div>"
        )

    disc = result.get("disclaimer", "")
    advice_html = f"""
    <div class='advice-stack'>
        {card('💊', 'Recommendations', result.get('recommendations', 'Not available.'))}
        {card('📋', 'Next Steps',      result.get('next_steps',      'Not available.'), 'card-steps')}
        {card('🌿', 'Self-Care Tips',  result.get('tips',            'Not available.'), 'card-tips')}
    </div>
    {"<div style='margin-top:10px;padding:8px 14px;background:rgba(83,11,14,0.05);border-radius:8px;font-size:0.74rem;color:var(--night-bordeaux);'>⚕ " + disc + "</div>" if disc else ""}
    <div class='time-footer'>⏱ Response generated in {elapsed:.2f}s</div>"""

    return circle_html, advice_html


with gr.Blocks(css=css, title="DermAI") as demo:

    
    gr.HTML(GEAR_HTML)

    gr.HTML("""
    <div class='derm-header'>
        <h1>🔬 DermAI <span class='pulse-dot'></span></h1>
        <p>Upload a skin lesion image for AI-powered classification and personalised medical guidance.</p>
    </div>""")

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_id="upload-col"):
            gr.HTML("<span class='panel-label'>📷 Image Input</span>")
            img_input = gr.Image(type="pil", label="", show_label=False)

            analyze_btn = gr.Button("✦ Analyze Image", elem_id="analyze-btn")

            gr.HTML("""
            <div id="progress-wrap">
                <div id="progress-inner"></div>
            </div>""")

            gr.HTML("<span class='panel-label' style='margin-top:18px;'>📊 Confidence</span>")
            confidence_out = gr.HTML(PH_CIRC)

        with gr.Column(scale=1, elem_id="results-col"):
            gr.HTML("<span class='panel-label'>📑 Analysis Results</span>")
            results_out = gr.HTML(PH_RES)

    gr.HTML("<div class='page-footer'>⚕ For educational purposes only. Always consult a qualified dermatologist.</div>")

    gr.HTML(LOADER_JS)

    analyze_btn.click(
        fn=analyze_with_ui,
        inputs=[img_input],
        outputs=[confidence_out, results_out],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)