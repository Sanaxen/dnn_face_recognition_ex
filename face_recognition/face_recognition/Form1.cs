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

            button1.Enabled = true;    //camera
            button2.Enabled = false;   //stop
            button3.Enabled = true;    //movie
            button4.Enabled = true;    //image
            button5.Enabled = true;    //add user
            button6.Enabled = false;    //movie
            button7.Enabled = false;    //image
            pictureBox1.AllowDrop = true;
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

                button1.Enabled = true;    //camera
                button2.Enabled = false;     //stop
                button3.Enabled = true;    //movie
                button4.Enabled = true;    //image
                button5.Enabled = true;    //add user
                button6.Enabled = true;    //movie
                button7.Enabled = true;    //image
            }
            string filename = cuDir + "\\break.run";
            try
            {
                if (System.IO.File.Exists(filename))
                {
                    System.IO.File.Delete(filename);
                }
            }
            catch { }

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

            string filename2 = cuDir + "\\break.run";
            try
            {
                if (System.IO.File.Exists(filename2))
                {
                    System.IO.File.Delete(filename2);
                }
            }
            catch { }
        }

        string solver()
        {
            if (radioButton1.Checked) return @"cuda\\dnn_face_recognition_ex.exe";
            if (radioButton2.Checked) return @"cpu\\dnn_face_recognition_ex.exe";
            if (radioButton3.Checked) return @"cpu\\dnn_face_recognition_ex_MKL.exe";

            return @"cpu\\dnn_face_recognition_ex.exe";
        }

        string detector()
        {
            if (radioButton6.Checked) return "--dnn_face_detect 0";
            if (radioButton5.Checked) return "--dnn_face_detect 1";
            if (radioButton4.Checked) return "--dnn_face_detect 2";

            return "--dnn_face_detect 0";
        }

        private void button1_Click(object sender, EventArgs e)
        {
            button1.Enabled = false;    //camera
            button2.Enabled = true;     //stop
            button3.Enabled = false;    //movie
            button4.Enabled = false;    //image
            button5.Enabled = false;    //add user
            button6.Enabled = false;    //movie
            button7.Enabled = false;    //image

            System.IO.Directory.SetCurrentDirectory(cuDir);
            pictureBox1.Image = null;
            app = new System.Diagnostics.ProcessStartInfo();

            clear();

            app.FileName = solver();
            app.Arguments = " " + "--no_show 1";
            app.Arguments += " " + "--num_gitters " + numericUpDown1.Value.ToString();
            app.Arguments += " " + "--t " + textBox1.Text.ToString();
            app.Arguments += " " + detector();
            app.Arguments += " " + "--camID " + numericUpDown1.Value.ToString();
            app.Arguments += " " + detector();
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
            if (System.IO.File.Exists("tmp\\end_image"))
            {
                timer1.Stop();
                button1.Enabled = true;    //camera
                button2.Enabled = false;     //stop
                button3.Enabled = true;    //movie
                button4.Enabled = true;    //image
                button5.Enabled = true;    //add user
                button6.Enabled = true;    //movie
                button7.Enabled = true;    //image
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            button1.Enabled = true;    //camera
            button2.Enabled = false;     //stop
            button3.Enabled = true;    //movie
            button4.Enabled = true;    //image
            button5.Enabled = true;    //add user
            button6.Enabled = true;    //movie
            button7.Enabled = true;    //image

            string filename = cuDir + "\\break.run";
            try
            {
                if (System.IO.File.Exists(filename))
                {
                    System.IO.File.Delete(filename);
                }
                var f = System.IO.File.Create(filename);
                f.Close();
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
            button6_Click(sender, e);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (openFileDialog2.ShowDialog() != DialogResult.OK)
            {
                return;
            }
            button7_Click(sender, e);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            if (openFileDialog2.ShowDialog() != DialogResult.OK)
            {
                return;
            }
            button1.Enabled = false;    //camera
            button2.Enabled = true;     //stop
            button3.Enabled = false;    //movie
            button4.Enabled = false;    //image
            button5.Enabled = true;    //add user
            button6.Enabled = false;    //movie
            button7.Enabled = false;    //image

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

            app.FileName = solver();
            app.Arguments = " " + "--num_gitters " + numericUpDown1.Value.ToString();
            app.Arguments += " " + openFileDialog2.FileName;
            app.UseShellExecute = false;

            timer1.Stop();
            pictureBox1.Image = CreateImage(openFileDialog2.FileName);
            process = System.Diagnostics.Process.Start(app);
            process.WaitForExit();

            button1.Enabled = true;    //camera
            button2.Enabled = false;     //stop
            button3.Enabled = true;    //movie
            button4.Enabled = true;    //image
            button5.Enabled = true;    //add user
            button6.Enabled = false;    //movie
            button7.Enabled = false;    //image
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            processClose(process);
        }

        private void timer2_Tick(object sender, EventArgs e)
        {

        }

        private void toggleEffectsToDragEvent(DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.All;
            else
                e.Effect = DragDropEffects.None;
        }
        private string getFileNameToDragEvent(DragEventArgs e)
        {
            string[] fileName = (string[])e.Data.GetData(DataFormats.FileDrop);
            if (System.IO.File.Exists(fileName[0]) == true)
            {
                return fileName[0];
            }
            else
            {
                return null;
            }
        }
        private void pictureBox1_DragEnter(object sender, DragEventArgs e)
        {
            toggleEffectsToDragEvent(e);
        }

        private void pictureBox1_DragDrop(object sender, DragEventArgs e)
        {
            string fileName = this.getFileNameToDragEvent(e);
            string ext = System.IO.Path.GetExtension(fileName);

            if ( ext == ".png" || ext == ".jpg")
            {
                openFileDialog2.FileName = fileName;
                button1.Enabled = false;    //camera
                button2.Enabled = true;     //stop
                button3.Enabled = false;    //movie
                button4.Enabled = true;    //image
                button5.Enabled = false;    //add user
                button6.Enabled = false;    //movie
                button7.Enabled = true;    //image
            }
            if (ext == ".mp4" || ext == ".avi")
            {
                openFileDialog1.FileName = fileName;
                button1.Enabled = false;    //camera
                button2.Enabled = true;     //stop
                button3.Enabled = true;    //movie
                button4.Enabled = false;    //image
                button5.Enabled = false;    //add user
                button6.Enabled = true;    //movie
                button7.Enabled = false;    //image
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.FileName == "") return;
            button1.Enabled = false;    //camera
            button2.Enabled = true;     //stop
            button3.Enabled = false;    //movie
            button4.Enabled = false;    //image
            button5.Enabled = false;    //add user

            System.IO.Directory.SetCurrentDirectory(cuDir);
            pictureBox1.Image = null;
            app = new System.Diagnostics.ProcessStartInfo();

            clear();

            app.FileName = solver();
            app.Arguments = " " + "--no_show 1";
            app.Arguments += " " + "--num_gitters " + numericUpDown1.Value.ToString();
            app.Arguments += " " + "--t " + textBox1.Text.ToString();
            app.Arguments += " " + detector();
            app.Arguments += " " + "--video";
            app.Arguments += " " + "\""+openFileDialog1.FileName+"\"";
            app.Arguments += " " + "--recog";
            app.UseShellExecute = true;
            app.WindowStyle = System.Diagnostics.ProcessWindowStyle.Minimized;

            timer1.Start();
            process = System.Diagnostics.Process.Start(app);
        }

        private void button7_Click(object sender, EventArgs e)
        {
            if (openFileDialog2.FileName == "") return;
            button1.Enabled = false;    //camera
            button2.Enabled = true;     //stop
            button3.Enabled = false;    //movie
            button4.Enabled = false;    //image
            button5.Enabled = false;    //add user

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

            app.FileName = solver();
            app.Arguments = " " + "--no_show 1";
            app.Arguments += " " + "--num_gitters " + numericUpDown1.Value.ToString();
            app.Arguments += " " + "--t " + textBox1.Text.ToString();
            app.Arguments += " " + detector();
            app.Arguments += " " + "--image";
            app.Arguments += " " + "\"" + openFileDialog2.FileName + "\"";
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
            button1.Enabled = true;    //camera
            button2.Enabled = false;     //stop
            button3.Enabled = true;    //movie
            button4.Enabled = true;    //image
            button5.Enabled = true;    //add user
            button6.Enabled = true;    //movie
            button7.Enabled = true;    //image
        }
    }
}
