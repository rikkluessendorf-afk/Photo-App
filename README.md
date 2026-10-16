# Photo Critic (Streamlit)

A password-protected photo critique app (Streamlit + Gemini), built to deploy free on
Streamlit Community Cloud and use from your phone.

Six categories scored 1-100. **Overall Execution = 50%** of the composite; the other five are 10% each.
Tiers: **S** 85+, **A** 75-84.9, **B** 65-74.9, **Field** below 65. Critiques are written in a teacher's
voice, and you can push back on any critique in a per-photo discussion. Every photo is judged in its own
fresh request, so there is no cheerleader drift over a long session.

## Files (all go in the root of your GitHub repo)

- `streamlit_app.py` : the app (screens, password gate)
- `critic_core.py` : rubric, scoring, Gemini calls (edit this to retune)
- `requirements.txt` : dependencies
- `README.md` : this file
- `anchors/` : optional, see "Calibrating" below

## Deploy (about 15 minutes, free)

**1. Get a Gemini API key.** Go to https://aistudio.google.com/apikey and create a key. Keep it private.

**2. Put the files on GitHub.**
1. Sign up at https://github.com (free) and choose *New repository*. Name it `photo-critic`.
2. Choose **Public** unless you plan to add anchor photos (see the privacy note below).
3. On the new repo page, choose *uploading an existing file*, drag in the four files above, and click
   *Commit changes*. Never put your API key or password in any file.

**3. Create the app on Streamlit Community Cloud.**
1. Go to https://share.streamlit.io and sign in with GitHub. Authorize access when asked.
2. Click **Create app**, then **Yup, I have an app**.
3. Pick your repository and branch (`main`). Set **Main file path** to `streamlit_app.py`.
4. Optionally choose a custom subdomain in **App URL** (for example `my-photo-critic`).
5. Click **Advanced settings**:
   - Set **Python version** to **3.12**.
   - In **Secrets**, paste this (with your real values), then click **Save**:
     ```toml
     GEMINI_API_KEY = "paste-your-gemini-key-here"
     APP_PASSWORD = "choose-a-long-password"
     ```
6. Click **Deploy**. The first build takes a few minutes.

**4. Use it on your phone.** Open your `https://<your-subdomain>.streamlit.app` address. Sign in, then add it
to your Home Screen (Share menu on iPhone; three-dot menu on Android). Tap the upload area to choose
photos from your library or take a new one.

**Updating later:** edit a file on GitHub (pencil icon, then Commit). The app updates automatically.
Secrets can be changed any time under the app's *Settings -> Secrets*.

The app refuses to open if `APP_PASSWORD` is missing, so it can never go live unprotected.
Repeated wrong passwords temporarily lock sign-in.

## Optional settings (add to Secrets)

```toml
GEMINI_MODEL = "gemini-flash-latest"   # any Gemini model that accepts images
TEMPERATURE = "0.2"                    # lower = more consistent scores
MAX_BATCH = "10"                       # new photos per run
MIN_SECONDS_BETWEEN_CALLS = "5"        # spacing to stay under free-tier per-minute limits
MAX_IMAGE_EDGE = "2000"                # photos are shrunk to this long side before sending
MAX_ANCHORS = "6"
```

## Calibrating with your example photos (optional)

Create an `anchors/` folder in the repo, add example photos, and add `anchors/anchors.json`:

```json
[
  {"file": "sunset_pier.jpg", "score": 52, "note": "Pleasant but conventional; flat light."},
  {"file": "market_portrait.jpg", "score": 78, "note": "Competent, commercially strong portrait."},
  {"file": "fog_valley.jpg", "score": 88, "note": "Distinctive vision, gallery level."}
]
```

`score` is the approximate composite you'd give it. They are reference points, not tier boundaries.
Use 3-6 photos spread across the tiers (each one is sent with every request).

**Privacy note:** in a public repo, anyone can see the anchor photos. If you want them private, make the
repo **private**. Streamlit allows only one private app at a time, and a private app also asks viewers to sign in
(you are the developer, so you can always open it).

## Retuning

Category weights and tier thresholds (`CATEGORIES`, `TIERS`) and the instructor's standards
(`SYSTEM_PROMPT`) are at the top of `critic_core.py`. If scores feel too harsh, too generous, or
clustered, adjust the "CALIBRATION RULES" or add anchors.

## Good to know

- **Sleeping:** apps with no traffic for 12 hours go to sleep. Open the address and tap
  *Yes, get this app back up!*
- **Results live in your session.** Refreshing the page or losing the connection clears them, so use
  *Download results (CSV)* for anything you want to keep. On a phone, keep the screen on and the browser in
  the foreground while a batch is running.
- **Limits:** there is no total upload cap, only Gemini's per-minute and per-day request limits, which reset.
  If the daily limit is hit, the app says so.
- **Free-tier privacy:** on Gemini's free tier, Google may use submitted photos and critiques to improve its
  products. Skip anything you wouldn't want used that way, or switch to a paid key.
- **Storage:** photos are held in memory for your session only and are not saved by the app.
- **If the build fails on `pillow-heif`** (iPhone HEIC support), delete that line from `requirements.txt`.
  iPhones usually convert photos to JPEG on upload anyway.
