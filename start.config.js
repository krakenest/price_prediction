module.exports = {
    apps: [{
    name: "sn50_price",
    script: "main.py",            // <-- your script, not /usr/bin/python3
    interpreter: "~/miniconda3/bin/python",   // <-- tell PM2 to use python
    args: "realtime --lookback-minutes 20 --grace-seconds 45",
    cron_restart: "2,12,22,32,42,52 * * * *", // 10*n + 2 minutes
    autorestart: false,                   // same as --no-autorestart
      time: true,
      env: {
        CMC_API_KEY: process.env.CMC_API_KEY,
        DATABASE_URL: process.env.DATABASE_URL
      }
    }]
}