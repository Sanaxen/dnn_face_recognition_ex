using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace face_recognition
{
    public partial class Form1 : Form
    {
        public static string MyPath = "";
        public static string cuDir = @"C:\Users\bccat\Desktop\dnn_face_recognition_ex\bin";
        System.Diagnostics.ProcessStartInfo app = null;
        System.Diagnostics.Process process = null;

        int image_count = 0;
        public Form1()
        {
            InitializeComponent();
            MyPath = System.AppDomain.CurrentDomain.BaseDirectory;
            cuDir = System.IO.Directory.GetCurrentDirectory();
        }

        public static System.Drawing.Image CreateImage(string filename)
        {
            System.Drawing.Image img = null;
            try
            {
                System.IO.FileStream fs = new System.IO.FileStream(
                    filename,
                    System.IO.FileMode.Open,
                    System.IO.FileAccess.Read);
                img = System.Drawing.Image.FromStream(fs);
                fs.Close();
            }
            catch
            {
            }
            return img;
        }

        void processClose(System.Diagnostics.Process process)
        {
            if (process == null) return;
            if (!process.HasExited)
            {
                process.Kill();
                //KillProcessTree(process);
                process = null;
            }
            timer1.Stop();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void clear()
        {
            for ( int i = 0; i < 10000; i++)
            {
                string filename = cuDir + "\\tmp\\tmp_" + $"{i:00000}" + ".png";
                if (System.IO.File.Exists(filename))
                {
                    System.IO.File.Delete(filename);
                }
            }

            int count = 10000;
            do
            {
                string filename = cuDir + "\\tmp\\tmp_" + $"{count:00000}" + ".png";
                if (System.IO.File.Exists(filename))
                {
                    System.IO.File.Delete(filename);
                    count++;
                }else
                {
                    break;
                }
            } while (true);
            image_count = 0;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            System.IO.Directory.SetCurrentDirectory(cuDir);
            pictureBox1.Image = null;
            app = new System.Diagnostics.ProcessStartInfo();

            clear();

            app.FileName = @"cuda\\dnn_face_recognition_ex.exe";
            app.Arguments = " " + "--no_show 1";
            app.Arguments += " " + "--camID " + numericUpDown1.Value.ToString();
            app.Arguments += " " + "--recog";
            app.UseShellExecute = true;
            app.WindowStyle = System.Diagnostics.ProcessWindowStyle.Minimized;

            timer1.Start();
            process = System.Diagnostics.Process.Start(app);
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            string filename = cuDir+"\\tmp\\tmp_" +$"{image_count:00000}" + ".png";
            if ( System.IO.File.Exists(filename))
            {
                try
                {
                    var tmp = pictureBox1.Image;
                    pictureBox1.Image = CreateImage(filename);
                    if (pictureBox1.Image == null)
                    {
                        pictureBox1.Image = tmp;
                    }
                    else
                    {
                        System.IO.File.Delete(filename);
                        image_count++;
                    }
                }
                catch { }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            string filename = cuDir + "\\break.run";
            try
            {
                if (System.IO.File.Exists(filename))
                {
                    System.IO.File.Delete(filename);
                }
                System.IO.File.Create(filename);
                //await System.Threading.Tasks.Task.Delay(2000);
                System.Threading.Thread.Sleep(2000);
            }
            catch { }

            processClose(process);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if ( openFileDialog1.ShowDialog() != DialogResult.OK)
            {
                return;
            }
            System.IO.Directory.SetCurrentDirectory(cuDir);
            pictureBox1.Image = null;
            app = new System.Diagnostics.ProcessStartInfo();

            clear();

            app.FileName = @"cuda\\dnn_face_recognition_ex.exe";
            app.Arguments = " " + "--no_show 1";
            app.Arguments += " " + "--video";
            app.Arguments += " " + openFileDialog1.FileName;
            app.Arguments += " " + "--recog";
            app.UseShellExecute = true;
            app.WindowStyle = System.Diagnostics.ProcessWindowStyle.Minimized;

            timer1.Start();
            process = System.Diagnostics.Process.Start(app);

        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (openFileDialog2.ShowDialog() != DialogResult.OK)
            {
                return;
            }
            System.IO.Directory.SetCurrentDirectory(cuDir);
            pictureBox1.Image = null;
            app = new System.Diagnostics.ProcessStartInfo();

            clear();
            string filename = cuDir + "\\output.png";
            if (System.IO.File.Exists(filename))
            { 
                System.IO.File.Delete(filename);
            }
            pictureBox1.Image = null;

            app.FileName = @"cuda\\dnn_face_recognition_ex.exe";
            app.Arguments = " " + "--no_show 1";
            app.Arguments += " " + "--image";
            app.Arguments += " " + openFileDialog2.FileName;
            app.UseShellExecute = true;
            app.WindowStyle = System.Diagnostics.ProcessWindowStyle.Minimized;

            timer1.Stop();
            process = System.Diagnostics.Process.Start(app);
            process.WaitForExit();

            if (System.IO.File.Exists(filename))
            {
                try
                {
                    var tmp = pictureBox1.Image;
                    pictureBox1.Image = CreateImage(filename);
                }
                catch { }
            }

        }

        private void button5_Click(object sender, EventArgs e)
        {
            if (openFileDialog2.ShowDialog() != DialogResult.OK)
            {
                return;
            }
            System.IO.Directory.SetCurrentDirectory(cuDir);
            pictureBox1.Image = null;
            app = new System.Diagnostics.ProcessStartInfo();

            clear();
            string filename = cuDir + "\\output.png";
            if (System.IO.File.Exists(filename))
            {
                System.IO.File.Delete(filename);
            }
            pictureBox1.Image = null;

            app.FileName = @"cuda\\dnn_face_recognition_ex.exe";
            app.Arguments = " " + openFileDialog2.FileName;
            app.UseShellExecute = false;

            timer1.Stop();
            pictureBox1.Image = CreateImage(openFileDialog2.FileName);
            process = System.Diagnostics.Process.Start(app);
            process.WaitForExit();
        }
    }
}
