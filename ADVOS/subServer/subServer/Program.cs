using System;
using System.IO;
using System.Collections;
using System.Linq;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace subServer
{
    class Server
    {
        private static FileInfo[] data_files = null;
        
        private static void FileInit()
        {
            string path = System.Environment.CurrentDirectory;
            //Console.WriteLine(path);
            DirectoryInfo info = new DirectoryInfo(path);
            var dir_list = info.GetDirectories();
            foreach (var item in dir_list)
            {
                if (item.Name == "data")
                {
                    DirectoryInfo data = new DirectoryInfo(item.FullName);
                    //Console.WriteLine(data.FullName);
                    data_files = data.GetFiles();
                }
            }
        }
        private static TcpListener listener { get; set; }
        private static int port;
        private static bool accept { get; set; } = false;
        public static void StartServer()
        {
            IPAddress address = IPAddress.Parse("127.0.0.1");
            listener = new TcpListener(address, port);
            listener.Start();
            accept = true;
            Console.WriteLine($"Server started. Listening to TCP clients at 127.0.0.1:{port}");
        }
        public static void Listen()
        {
            if (listener != null && accept)
            {
                while (true)
                {
                    Console.WriteLine("Waiting for client...");
                    var clientTask = listener.AcceptTcpClientAsync();
                    if (clientTask.Result != null)
                    {
                        Console.WriteLine("Client connected. Waiting for data.");
                        var client = clientTask.Result;
                        string message = "";
                        byte[] data = Encoding.UTF8.GetBytes("send request");
                        client.GetStream().Write(data, 0, data.Length);
                        byte[] buffer = new byte[1024];
                        client.GetStream().Read(buffer, 0, buffer.Length);
                        message = Encoding.UTF8.GetString(buffer);
                        Console.WriteLine(message);
                        if (message.StartsWith("getfilelist"))
                        {
                            SendFileList(client, out message, out data, out buffer);
                        }
                        
                    }

                }
            }
        }

        private static void SendFileList(TcpClient client, out string message, out byte[] data, out byte[] buffer)
        {
            foreach (var file in data_files)
            {
                data = Encoding.UTF8.GetBytes(file.Name);
                client.GetStream().Write(data, 0, data.Length);
                buffer = new byte[1024];
                client.GetStream().Read(buffer, 0, buffer.Length);
            }
            data = Encoding.UTF8.GetBytes("finished");
            client.GetStream().Write(data, 0, data.Length);
            buffer = new byte[1024];
            client.GetStream().Read(buffer, 0, buffer.Length);
            message = Encoding.UTF8.GetString(buffer);
            if (message.StartsWith("quit"))
            {
                Console.WriteLine("Closing connection.");
                client.GetStream().Dispose();
            }
        }

        public static void SendFileInfo()
        {

        }
        static void Main(string[] args)
        {
            FileInit();
            foreach (var file in data_files)
            {
                Console.WriteLine(file.FullName);
            }
            //Console.Read();
            port = 1080;
            Server.StartServer();
            Server.Listen(); // Start listening.  
            
        }
    }
}
